import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, List, Tuple


# ---------------------------------------------------------------------------
# Class merge map — applied consistently across ALL data loading functions
#
# Cosine similarity analysis on N-BaIoT features confirmed:
#   Class 4 (Mirai_udp) vs Class 5 (Mirai_udpplain) : similarity = 1.000
#   Class 6 (Gafgyt_combo) vs Class 9 (Gafgyt_tcp)  : similarity = 0.999
# These pairs are physically indistinguishable in feature space.
# Merge 5→4 and 9→6, then remap to contiguous 0-8 (9 classes total).
#
# Original 11 classes → Merged 9 classes:
#   0  Benign          → 0  Benign
#   1  Mirai_ack       → 1  Mirai_ack
#   2  Mirai_scan      → 2  Mirai_scan
#   3  Mirai_syn       → 3  Mirai_syn
#   4  Mirai_udp       → 4  Mirai_udp  (absorbs class 5)
#   5  Mirai_udpplain  → 4  (merged into Mirai_udp)
#   6  Gafgyt_combo    → 5  Gafgyt_combo (absorbs class 9)
#   7  Gafgyt_junk     → 6  Gafgyt_junk
#   8  Gafgyt_scan     → 7  Gafgyt_scan
#   9  Gafgyt_tcp      → 5  (merged into Gafgyt_combo)
#   10 Gafgyt_udp      → 8  Gafgyt_udp
# ---------------------------------------------------------------------------

MERGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
             5: 4, 6: 5, 7: 6, 8: 7, 9: 5, 10: 8}

NUM_CLASSES = 9

CLASS_NAMES = [
    "Benign",        # 0
    "Mirai_ack",     # 1
    "Mirai_scan",    # 2
    "Mirai_syn",     # 3
    "Mirai_udp",     # 4  (merged with Mirai_udpplain)
    "Gafgyt_combo",  # 5  (merged with Gafgyt_tcp)
    "Gafgyt_junk",   # 6
    "Gafgyt_scan",   # 7
    "Gafgyt_udp",    # 8
]


def apply_merge(y: np.ndarray) -> np.ndarray:
    """Apply MERGE_MAP to a label array. Used consistently everywhere."""
    return np.array([MERGE_MAP[int(lbl)] for lbl in y], dtype=np.int64)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class NBaiotDataset(Dataset):
    """
    Wraps (features, labels) numpy arrays as a PyTorch Dataset.
    float32 required by model forward pass.
    long  required by nn.CrossEntropyLoss.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels,   dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------

class DataManager:
    def __init__(
        self,
        data_dir: str,
        sample_size_per_file: Optional[int] = 50000,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        """
        data_dir             : path to partitioned client data folder
        sample_size_per_file : max rows loaded per client CSV (None = all)
        train_ratio          : 80% train / 20% test split per client
        seed                 : RNG seed for reproducible splits
        """
        self.data_dir    = data_dir
        self.sample_size = sample_size_per_file
        self.train_ratio = train_ratio
        self.seed        = seed

        self.label_map: Dict[int, int] = self._build_global_label_map()
        self.scaler = StandardScaler()
        self._fit_global_scaler()
        self.selected_features: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Label map — built globally across all clients
    # ------------------------------------------------------------------

    def _build_global_label_map(self) -> Dict[int, int]:
        label_set = set()
        for file in sorted(os.listdir(self.data_dir)):
            if file.startswith("client_") and file.endswith(".csv"):
                path = os.path.join(self.data_dir, file)
                try:
                    df = pd.read_csv(path, nrows=100)
                    label_set.update(df.iloc[:, -1].unique())
                except Exception as e:
                    print(f"[DataManager] Warning reading {file}: {e}")

        sorted_labels = sorted(list(label_set))
        label_map = {lbl: idx for idx, lbl in enumerate(sorted_labels)}
        print(f"[DataManager] Label map: {len(label_map)} classes: {label_map}")
        return label_map

    def get_num_clients(self) -> int:
        return len([f for f in os.listdir(self.data_dir)
                    if f.startswith("client_") and f.endswith(".csv")])

    def get_client_ids(self) -> List[int]:
        ids = []
        for f in os.listdir(self.data_dir):
            if f.startswith("client_") and f.endswith(".csv"):
                try:
                    ids.append(int(f.replace("client_", "").replace(".csv", "")))
                except ValueError:
                    pass
        return sorted(ids)

    # ------------------------------------------------------------------
    # Global scaler — fit on sample of all client files
    # ------------------------------------------------------------------

    def _fit_global_scaler(self, rows_per_file: int = 2000):
        print("[DataManager] Fitting global scaler...")
        sample_chunks = []

        for file in sorted(os.listdir(self.data_dir)):
            if file.startswith("client_") and file.endswith(".csv"):
                path = os.path.join(self.data_dir, file)
                try:
                    df = pd.read_csv(path, nrows=rows_per_file)
                    sample_chunks.append(df.iloc[:, :-1].values)
                except Exception as e:
                    print(f"[DataManager] Warning reading {file}: {e}")

        if not sample_chunks:
            raise RuntimeError(
                f"No client CSV files found in {self.data_dir}. "
                f"Run partition_data.py first."
            )

        combined = np.vstack(sample_chunks)
        self.scaler.fit(combined)
        print(f"[DataManager] Scaler fitted on {len(combined):,} rows "
              f"across {len(sample_chunks)} files.")

    # ------------------------------------------------------------------
    # Feature selection hook — called after Algorithm 1
    # ------------------------------------------------------------------

    def apply_feature_selection(self, selected_indices: np.ndarray):
        """
        Store selected feature indices from Algorithm 1.
        Must be called BEFORE any load_client_data() calls.
        """
        self.selected_features = selected_indices
        print(f"[DataManager] Feature selection applied — "
              f"{len(selected_indices)} features selected.")

    # ------------------------------------------------------------------
    # Internal: load, scale, feature-select one client's full data
    # ------------------------------------------------------------------

    def _load_full(self, client_id: int) -> Tuple[np.ndarray, np.ndarray]:
        path = os.path.join(self.data_dir, f"client_{client_id}.csv")
        if not os.path.exists(path):
            raise ValueError(
                f"Client file not found: {path}. Run partition_data.py first."
            )

        df    = pd.read_csv(path, nrows=self.sample_size)
        X     = df.iloc[:, :-1].values.astype(np.float32)
        y_raw = df.iloc[:, -1].values
        y     = np.array([self.label_map[val] for val in y_raw], dtype=np.int64)

        # Merge indistinguishable classes (confirmed by cosine similarity = 1.00)
        y = apply_merge(y)

        # Scale on all features
        X = self.scaler.transform(X)

        # Slice to selected features (after Algorithm 1)
        if self.selected_features is not None:
            X = X[:, self.selected_features]

        return X, y

    # ------------------------------------------------------------------
    # Deterministic 80/20 split per client
    # ------------------------------------------------------------------

    def _split(
        self, X: np.ndarray, y: np.ndarray, client_id: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n     = len(X)
        rng   = np.random.default_rng(self.seed + client_id)
        idx   = rng.permutation(n)
        split = int(n * self.train_ratio)
        return (
            X[idx[:split]], y[idx[:split]],
            X[idx[split:]], y[idx[split:]],
        )

    # ------------------------------------------------------------------
    # Public API: training data (80% per client)
    # ------------------------------------------------------------------

    def load_client_data(self, client_id: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = self._load_full(client_id)
        X_train, y_train, _, _ = self._split(X, y, client_id)
        return X_train, y_train

    # ------------------------------------------------------------------
    # Public API: client local test data (20% per client)
    # NOT used for global accuracy — kept for optional local diagnostics
    # ------------------------------------------------------------------

    def load_client_test_data(self, client_id: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = self._load_full(client_id)
        _, _, X_test, y_test = self._split(X, y, client_id)
        return X_test, y_test

    # ------------------------------------------------------------------
    # Public API: global balanced test set
    # Used for unbiased round-by-round accuracy measurement
    # ------------------------------------------------------------------

    def load_global_test_data(self, n_per_class: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load globally balanced held-out test set.
        Applies class merge and samples n_per_class per class.
        n_per_class=1000 → 9000 total samples across 9 classes.
        Same seed every call — identical test set every round.
        """
        path = os.path.join(self.data_dir, "global_test.npz")
        if not os.path.exists(path):
            raise RuntimeError(
                f"Global test set not found at {path}.\n"
                f"Re-run partition_data.py to generate it."
            )

        data  = np.load(path)
        X_raw = data['X'].astype(np.float32)
        y_raw = data['y'].astype(np.int64)

        # Apply same merge as training data
        y = apply_merge(y_raw)

        # Balance: sample n_per_class from each class
        rng = np.random.default_rng(42)
        balanced_idx = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            n       = min(n_per_class, len(cls_idx))
            chosen  = rng.choice(cls_idx, size=n, replace=False)
            balanced_idx.extend(chosen.tolist())

        balanced_idx = np.array(balanced_idx)
        rng.shuffle(balanced_idx)

        X_raw = X_raw[balanced_idx]
        y     = y[balanced_idx]

        X = self.scaler.transform(X_raw)
        if self.selected_features is not None:
            X = X[:, self.selected_features]

        print(f"[DataManager] Global test set loaded — "
              f"{len(X):,} samples ({n_per_class} per class, "
              f"{len(np.unique(y))} classes).")
        return X, y

    # ------------------------------------------------------------------
    # Public API: raw unscaled data — for Algorithm 1 Chi-square
    # ------------------------------------------------------------------

    def load_client_data_raw(self, client_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Raw unscaled features for Chi-square computation.
        No scaling, no feature selection, full dataset (no split).
        Merge IS applied so chi-square scores match 9-class training setup.
        """
        path = os.path.join(self.data_dir, f"client_{client_id}.csv")
        if not os.path.exists(path):
            raise ValueError(f"Client file not found: {path}")

        df    = pd.read_csv(path)
        X_raw = df.iloc[:, :-1].values.astype(np.float32)
        y_raw = df.iloc[:, -1].values
        y     = np.array([self.label_map[val] for val in y_raw], dtype=np.int64)

        # Apply merge — chi-square must see same 9 classes as training
        y = apply_merge(y)

        return X_raw, y
