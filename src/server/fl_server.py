import numpy as np
import torch
import flwr as fl
from flwr.common import (
    Context, Parameters, FitRes, EvaluateRes, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional, Tuple, Union

from src.strategies.aggregation import HBFLAggregator
from src.models.conv_mlp import build_model
from src.utils.data_loader import NBaiotDataset


class HBFLStrategy(fl.server.strategy.Strategy):
    """
    HB-FL-IDS Full Strategy — Algorithms 3, 4, 5.

    Features:
        1. Uses HBFLAggregator with full defense:
           - Adaptive trimming, double-sided trimming
           - Layer-wise MZ-score + PDP filtering (LASA)
           - Combined trust score: cosine * layer_frac - div_penalty
           - Threat-reactive EMA reputation
           - w^{t+1} = w^t - eta_g * g^t (gradient DESCENT)

        2. Server-side evaluation on global_test.npz every round:
           - Same balanced test set each round — comparable accuracy
           - Byzantine clients completely excluded from metric
           - Uniform class distribution — no imbalance distortion

        3. Passes layer_shapes to aggregator for layer-wise processing.
           layer_shapes extracted from server model via get_layer_shapes().

        4. cid_to_idx maps Flower string cids to stable integer indices
           so reputation accumulates correctly per client across rounds
           even when different subsets participate each round.
    """

    def __init__(
        self,
        num_clients: int,
        global_test_dataset,
        input_dim: int = 70,
        num_classes: int = 9,
        global_lr: float = 0.1,
        fraction_fit: float = 0.2,
        min_fit_clients: int = 10,
        byzantine_fraction: float = 0.2,
        beta: float = 0.9,
        theta_min: float = 0.0,
        gamma_0: float = 0.1,
        lambda_adapt: float = 0.1,
        gamma_min: float = 0.05,
        lambda_m: float = 3.5,
        lambda_d: float = 3.5,
    ):
        super().__init__()
        self.num_clients         = num_clients
        self.global_lr           = global_lr
        self.fraction_fit        = fraction_fit
        self.min_fit_clients     = min_fit_clients
        self.global_test_dataset = global_test_dataset

        # Server-side model — same architecture as clients
        self.model = build_model(input_dim=input_dim, num_classes=num_classes)
        self.d     = self.model.get_gradient_dim()
        self.global_weights = self._flatten_model()

        # Extract layer_shapes for HBFLAggregator layer-wise processing
        layer_shapes = self.model.get_layer_shapes()

        self.aggregator = HBFLAggregator(
            num_clients=num_clients,
            layer_shapes=layer_shapes,
            beta=beta,
            theta_min=theta_min,
            gamma_0=gamma_0,
            lambda_adapt=lambda_adapt,
            gamma_min=gamma_min,
            byzantine_fraction=byzantine_fraction,
            lambda_m=lambda_m,
            lambda_d=lambda_d,
        )
        # from old_hbfl_aggregator import OldHBFLAggregator

        # self.aggregator = OldHBFLAggregator(
        #     num_clients=num_clients,
        #     byzantine_fraction=byzantine_fraction,
        # )

        self.round_metrics: List[Dict] = []
        self.cid_to_idx: Dict[str, int] = {}
        self.next_client_id: int = 0

        print(f"\n[Server] HBFLStrategy ready — "
              f"N={num_clients}, d={self.d:,}, "
              f"fraction_fit={fraction_fit} "
              f"({int(num_clients * fraction_fit)} clients/round)")

    # ------------------------------------------------------------------
    # Flower Strategy interface
    # ------------------------------------------------------------------

    def initialize_parameters(
        self, client_manager: fl.server.ClientManager
    ) -> Optional[Parameters]:
        """Algorithm 5 Line 7: broadcast w^0 (random initialization)."""
        print("[Server] Broadcasting initial model w^0.")
        return self._weights_to_parameters(self.global_weights)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        sample_size = max(
            self.min_fit_clients,
            int(self.num_clients * self.fraction_fit),
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients,
        )
        print(f"\n[Server] Round {server_round} — "
              f"selected {len(clients)}/{self.num_clients} clients")
        fit_ins = fl.common.FitIns(parameters, {"server_round": server_round})
        return [(c, fit_ins) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            print("[Server] No results — skipping round.")
            return None, {}

        if failures:
            print(f"[Server] {len(failures)} client(s) failed.")

        client_gradients: List[np.ndarray] = []
        participating_ids: List[int] = []

        for client_proxy, fit_res in results:
            param_arrays = parameters_to_ndarrays(fit_res.parameters)
            g_hat = np.concatenate([arr.ravel() for arr in param_arrays])
            client_gradients.append(g_hat)

            # Map Flower cid -> stable integer index
            cid = client_proxy.cid
            if cid not in self.cid_to_idx:
                self.cid_to_idx[cid] = self.next_client_id
                self.next_client_id += 1
            participating_ids.append(self.cid_to_idx[cid])

        g_hats = np.vstack(client_gradients)

        new_weights, agg_metrics = self.aggregator.aggregate(
            g_hats=g_hats,
            participating_ids=participating_ids,
            global_weights=self.global_weights,
            global_lr=self.global_lr,
        )
        self.global_weights = new_weights
        self._load_weights_into_model(new_weights)

        suspicious = self.aggregator.get_suspicious_clients(threshold=0.1)
        if suspicious:
            print(f"[Server] Suspicious clients: "
                  f"{suspicious[:10]}{'...' if len(suspicious) > 10 else ''}")

        self.round_metrics.append({
            "server_round":    server_round,
            "n_participated":  len(results),
            "participating_ids": participating_ids,
            **agg_metrics,
        })

        return self._weights_to_parameters(new_weights), {
            "gamma_t":         float(agg_metrics.get("gamma_t", 0.0)),
            "mean_reputation": float(agg_metrics.get("mean_reputation", 1.0)),
            "min_reputation":  float(agg_metrics.get("min_reputation", 1.0)),
            "n_retained":      int(agg_metrics.get("n_retained", len(results))),
        }

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        # Server-side evaluation handles everything — no client evaluation
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return None, {}

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Server-side evaluation on globally balanced held-out test set.
        Called by Flower after each aggregate_fit.
        Uses self.model which is updated in aggregate_fit via _load_weights_into_model.
        """
        if self.global_test_dataset is None:
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        from torch.utils.data import DataLoader
        loader = DataLoader(
            self.global_test_dataset, batch_size=512, shuffle=False
        )

        correct, total = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X      = batch_X.to(device)
                batch_y      = batch_y.to(device)
                outputs      = self.model(batch_X)
                _, predicted = torch.max(outputs, 1)
                total   += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        loss     = 1.0 - accuracy

        print(f"[Server] Round {server_round} — "
              f"GLOBAL accuracy: {accuracy:.4f} ({total:,} samples)")

        if self.round_metrics:
            self.round_metrics[-1]["global_accuracy"] = float(accuracy)

        return float(loss), {"global_accuracy": float(accuracy)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten_model(self) -> np.ndarray:
        return np.concatenate([
            p.detach().cpu().numpy().ravel()
            for p in self.model.parameters() if p.requires_grad
        ])

    def _load_weights_into_model(self, flat_weights: np.ndarray):
        offset = 0
        with torch.no_grad():
            for p in self.model.parameters():
                if p.requires_grad:
                    size  = p.numel()
                    chunk = flat_weights[offset:offset + size]
                    p.copy_(torch.tensor(
                        chunk.reshape(p.shape), dtype=torch.float32
                    ))
                    offset += size

    def _weights_to_parameters(self, flat_weights: np.ndarray) -> Parameters:
        arrays, offset = [], 0
        for p in self.model.parameters():
            if p.requires_grad:
                size = p.numel()
                arrays.append(
                    flat_weights[offset:offset + size].reshape(p.shape)
                )
                offset += size
        return ndarrays_to_parameters(arrays)

    def get_round_metrics(self) -> List[Dict]:
        return self.round_metrics

    def get_final_weights(self) -> np.ndarray:
        return self.global_weights.copy()


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(
    data_manager,
    n_clients: int,
    byzantine_assignments: Dict,
    num_rounds: int = 100,
    fraction_fit: float = 0.2,
    global_lr: float = 0.5,
    byzantine_fraction: float = 0.2,
    theta_min: float = 0.0,
    input_dim: int = 70,
    num_classes: int = 11,
    local_epochs: int = 5,
    local_lr: float = 0.01,
    batch_size: int = 256,
    top_k_ratio: float = 0.3,
    residual_decay: float = 0.9,
    beta: float = 0.9,
    gamma_0: float = 0.1,
    lambda_adapt: float = 0.1,
    gamma_min: float = 0.05,
    lambda_m: float = 1.5,
    lambda_d: float = 1.5,
) -> Tuple:
    from src.clients.fl_client import NBaiotClient

    X_test, y_test      = data_manager.load_global_test_data()
    global_test_dataset = NBaiotDataset(X_test, y_test)

    strategy = HBFLStrategy(
        num_clients=n_clients,
        global_test_dataset=global_test_dataset,
        input_dim=input_dim,
        num_classes=num_classes,
        global_lr=global_lr,
        fraction_fit=fraction_fit,
        min_fit_clients=max(10, int(n_clients * fraction_fit)),
        byzantine_fraction=byzantine_fraction,
        beta=beta,
        theta_min=theta_min,
        gamma_0=gamma_0,
        lambda_adapt=lambda_adapt,
        gamma_min=gamma_min,
        lambda_m=lambda_m,
        lambda_d=lambda_d,
    )

    def client_fn(context: Context) -> fl.client.Client:
        client_id = int(context.node_config.get("partition-id", 0)) % n_clients

        X_train, y_train = data_manager.load_client_data(client_id)
        train_dataset    = NBaiotDataset(X_train, y_train)

        is_byz = client_id in byzantine_assignments
        attack = byzantine_assignments.get(client_id, None)

        client = NBaiotClient(
            client_id=client_id,
            dataset=train_dataset,
            input_dim=input_dim,
            num_classes=num_classes,
            local_epochs=local_epochs,
            local_lr=local_lr,
            batch_size=batch_size,
            is_byzantine=is_byz,
            attack_type=attack,
            top_k_ratio=top_k_ratio,
            residual_decay=residual_decay,
        )
        return client.to_client()

    print(f"\n[Simulation] HB-FL-IDS — "
          f"N={n_clients}, T={num_rounds}, "
          f"fraction_fit={fraction_fit} "
          f"({int(n_clients * fraction_fit)} clients/round), "
          f"Byzantine={len(byzantine_assignments)} ({byzantine_fraction*100:.0f}%), "
          f"rho={top_k_ratio}, xi={residual_decay}")

    import ray
    ray.init(
        num_cpus=4,
        object_store_memory=512 * 1024 * 1024,
        ignore_reinit_error=True,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.0},
    )

    return history, strategy.get_round_metrics(), strategy.get_final_weights()
