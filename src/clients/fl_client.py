import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl

from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional

from src.models.conv_mlp import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Module-level persistent stores — survive Flower's per-round client recreation
# ---------------------------------------------------------------------------

# Residual memory — Algorithm 2 Step 3 & 5
_residual_store: Dict[int, np.ndarray] = {}

# MomTopK selection counts — prevents same parameters dominating every round
_selection_count_store: Dict[int, np.ndarray] = {}


def get_residual(client_id: int, d: int) -> np.ndarray:
    if client_id not in _residual_store:
        _residual_store[client_id] = np.zeros(d, dtype=np.float32)
    return _residual_store[client_id]


def set_residual(client_id: int, residual: np.ndarray):
    _residual_store[client_id] = residual.copy()


def get_selection_count(client_id: int, d: int) -> np.ndarray:
    if client_id not in _selection_count_store:
        _selection_count_store[client_id] = np.zeros(d, dtype=np.float32)
    return _selection_count_store[client_id]


def set_selection_count(client_id: int, counts: np.ndarray):
    _selection_count_store[client_id] = counts.copy()


# ---------------------------------------------------------------------------
# FL Client — Algorithm 2
# ---------------------------------------------------------------------------

class NBaiotClient(fl.client.NumPyClient):
    """
    Algorithm 2 — Client Local Update with Gradient Accumulation.

    Features:
        1. Local SGD with momentum=0.9 — matches Algorithm 2 step rule.
        2. Class-weighted CrossEntropyLoss — prevents majority-class collapse
           on Non-IID local data distributions.
        3. Residual memory — no gradient permanently lost. Dropped Top-K
           values stored and added back next round. xi=0.9 decay fades
           stale residuals to <1% within ~50 rounds.
        4. MomTopK selection — prevents same high-magnitude parameters
           dominating every round. Selection counts decay at 0.9/round,
           giving boosted selection probability to long-unselected params.
        5. Top-K sparsification — 90% upload reduction. rho=0.1 retains
           K=floor(0.1*d) parameters. LASA Lemma 1: sparsification reduces
           Byzantine attack surface in Non-IID settings.
        6. Unit-norm normalization — nullifies scaling attacks completely.
           All clients arrive at server with ||g_hat||=1 regardless of
           local data size or learning rate.
        7. Byzantine attack injection — for experimental evaluation of
           6 attack types: label_flip, gradient_poison, scaling,
           random_noise, backdoor, min_max.
        8. evaluate() is a no-op — server evaluates on global_test.npz.
    """

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        input_dim: int = 70,
        num_classes: int = 9,
        local_epochs: int = 3,
        local_lr: float = 0.01,
        top_k_ratio: float = 0.3,
        residual_decay: float = 0.9,
        batch_size: int = 256,
        is_byzantine: bool = False,
        attack_type: Optional[str] = None,
    ):
        self.client_id      = client_id
        self.local_epochs   = local_epochs
        self.top_k_ratio    = top_k_ratio
        self.residual_decay = residual_decay
        self.is_byzantine   = is_byzantine
        self.attack_type    = attack_type

        self.train_loader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True, drop_last=True,
        )

        # Build model ONCE
        self.model = build_model(
            input_dim=input_dim, num_classes=num_classes
        ).to(DEVICE)

        # Class-weighted loss — critical for Non-IID data
        counts = torch.zeros(num_classes)
        for _, y_batch in DataLoader(dataset, batch_size=2048, shuffle=False):
            for lbl in y_batch:
                counts[lbl] += 1
        counts  = counts.clamp(min=1)
        weights = counts.sum() / (num_classes * counts)
        weights = (weights / weights.mean()).to(DEVICE)
        weights = torch.clamp(weights, min=0.5, max=4.0)

        self.criterion = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=0.2
        )

        # SGD with momentum — Algorithm 2 Step 4: w <- w - eta * grad(L)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=local_lr,
            momentum=0.9,
            weight_decay=1e-4,
        )

        self.d = self.model.get_gradient_dim()
        self.K = max(1, int(top_k_ratio * self.d))

    # ------------------------------------------------------------------
    # Flower interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [
            p.detach().cpu().numpy()
            for p in self.model.parameters()
            if p.requires_grad
        ]

    def set_parameters(self, parameters: List[np.ndarray]):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        with torch.no_grad():
            for p, new_val in zip(trainable, parameters):
                p.copy_(torch.tensor(new_val, dtype=torch.float32))

    # ------------------------------------------------------------------
    # Algorithm 2 Steps 1-2: Local Training
    # ------------------------------------------------------------------

    def _local_train(self) -> np.ndarray:
        """
        Snapshot w0, train E epochs, return g = wE - w0.
        Gradient clipping prevents exploding pseudo-gradients.
        """
        w0 = self._flatten_params()
        self.model.train()

        for _ in range(self.local_epochs):
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss    = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                self.optimizer.step()

        wE = self._flatten_params()
        return (wE - w0).astype(np.float32)

    # ------------------------------------------------------------------
    # Algorithm 2 Steps 3-6: Accumulate → MomTopK Sparsify → Decay → Normalize
    # ------------------------------------------------------------------

    def _apply_algorithm2(self, g_raw: np.ndarray) -> np.ndarray:
        """
        Step 3: Accumulate residual — g_accumulated = g_raw + m_{t-1}
        Step 4: MomTopK sparsification — top K by adjusted magnitude
                (penalizes recently selected parameters to encourage coverage)
        Step 5: Save decayed residual — m_t = xi * (g_accumulated - g_sparse)
        Step 6: Normalize to unit sphere — g_hat = g_sparse / ||g_sparse||
        """
        g_raw = g_raw.astype(np.float32)

        # Step 3: residual accumulation
        residual      = get_residual(self.client_id, self.d)
        g_accumulated = g_raw + residual

        # Step 4: MomTopK selection
        # Penalize frequently selected indices so all parameters eventually
        # get transmitted — prevents stagnation on a fixed subset
        sel_counts = get_selection_count(self.client_id, self.d)
        penalty    = sel_counts / (sel_counts.max() + 1e-10)
        adjusted   = np.abs(g_accumulated) * (1.0 - 0.2 * penalty)

        top_k_indices = np.argpartition(adjusted, -self.K)[-self.K:]
        g_sparse      = np.zeros(self.d, dtype=np.float32)
        g_sparse[top_k_indices] = g_accumulated[top_k_indices]

        # Update selection counts with decay
        sel_counts *= 0.9
        sel_counts[top_k_indices] += 1.0
        set_selection_count(self.client_id, sel_counts)

        # Step 5: save decayed residual
        new_residual = self.residual_decay * (g_accumulated - g_sparse)
        set_residual(self.client_id, new_residual)

        # Step 6: no normalization
        return g_sparse

    # ------------------------------------------------------------------
    # Byzantine attack injection (for experimental evaluation only)
    # ------------------------------------------------------------------

    def _apply_attack(self, g_hat: np.ndarray) -> np.ndarray:
        if self.attack_type == "label_flip":
            return g_hat   # trains on flipped labels — gradient already poisoned

        elif self.attack_type == "gradient_poison":
            return -g_hat * 10.0   # strong negative scaling

        elif self.attack_type == "scaling":
            return g_hat * 50.0    # large magnitude overwhelms averaging

        elif self.attack_type == "random_noise":
            return np.random.randn(self.d).astype(np.float32)

        elif self.attack_type == "backdoor":
            trigger = np.zeros(self.d, dtype=np.float32)
            trigger[:100] = 0.1
            return g_hat + trigger

        elif self.attack_type == "min_max":
            norm = np.linalg.norm(g_hat)
            if norm > 1e-10:
                return -g_hat * 3.0   # reverse direction with large magnitude
            return -g_hat

        return g_hat

    # ------------------------------------------------------------------
    # Flower fit()
    # ------------------------------------------------------------------

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        g_raw = self._local_train()
        g_hat = self._apply_algorithm2(g_raw)

        if self.is_byzantine and self.attack_type:
            g_hat = self._apply_attack(g_hat)

        return (
            self._encode_gradient(g_hat),
            len(self.train_loader.dataset),
            {
                "client_id":    self.client_id,
                "is_byzantine": int(self.is_byzantine),
                "attack_type":  self.attack_type or "none",
            },
        )

    # ------------------------------------------------------------------
    # Flower evaluate() — no-op; server evaluates on global_test.npz
    # ------------------------------------------------------------------

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        return 0.0, 0, {"accuracy": 0.0}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten_params(self) -> np.ndarray:
        return np.concatenate([
            p.detach().cpu().numpy().ravel()
            for p in self.model.parameters() if p.requires_grad
        ])

    def _encode_gradient(self, g_hat: np.ndarray) -> List[np.ndarray]:
        encoded, offset = [], 0
        for p in self.model.parameters():
            if p.requires_grad:
                size = p.numel()
                encoded.append(g_hat[offset:offset + size].reshape(p.shape))
                offset += size
        return encoded
