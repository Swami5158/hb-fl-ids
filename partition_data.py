"""
partition_data.py — Dirichlet Data Partitioner with Global Test Set
=====================================================================
Changes from original:
    1. Extracts a globally balanced 20% test set BEFORE partitioning.
       Each class contributes equally to the test set — uniform distribution.
       This gives an unbiased, comparable accuracy metric every round.
    2. Saves global_test.npz alongside client CSVs.
    3. Saves client_distribution.json for paper Non-IID analysis.
    4. Remaining 80% of data is partitioned across clients using Dirichlet.

Run ONCE before any FL experiment:
    python partition_data.py --n-clients 200 --alpha 0.5
    python partition_data.py --n-clients 200 --alpha 0.1
    python partition_data.py --n-clients 200 --alpha 1.0
    python partition_data.py --all-alphas --n-clients 200

Output structure:
    data/partitioned/alpha_0.5_N200/
        client_0.csv ... client_199.csv   <- 80% data, Dirichlet partitioned
        global_test.npz                   <- 20% data, uniform class balance
        partition_stats.json              <- Non-IID entropy stats
        client_distribution.json          <- per-client class counts for paper
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------

LABEL_NAMES = [
    "benign",
    "gafgyt.combo",
    "gafgyt.junk",
    "gafgyt.scan",
    "gafgyt.tcp",
    "gafgyt.udp",
    "mirai.ack",
    "mirai.scan",
    "mirai.syn",
    "mirai.udp",
    "mirai.udpplain",
]

LABEL_MAP = {name: idx for idx, name in enumerate(LABEL_NAMES)}
NUM_CLASSES = len(LABEL_NAMES)


# ---------------------------------------------------------------------------
# Step 1: Load raw CSVs
# ---------------------------------------------------------------------------

def load_raw_data(raw_dir: str, samples_per_file: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
    print(f"\n[Partitioner] Loading raw data from {raw_dir}...")
    all_X, all_y = [], []

    for file in sorted(os.listdir(raw_dir)):
        if not (file.endswith(".csv") and file[0].isdigit()):
            continue

        parts = file.split(".")
        label_name = ".".join(parts[1:-1])

        if label_name not in LABEL_MAP:
            print(f"  [Warning] Unknown label '{label_name}' in {file} — skipping.")
            continue

        label_id = LABEL_MAP[label_name]
        path = os.path.join(raw_dir, file)

        try:
            df = pd.read_csv(path, nrows=samples_per_file)
            X  = df.values.astype(np.float32)
            y  = np.full(len(df), label_id, dtype=np.int64)
            all_X.append(X)
            all_y.append(y)
            print(f"  Loaded {file}: {len(df):,} samples, label={label_name}({label_id})")
        except Exception as e:
            print(f"  [Warning] Could not read {file}: {e}")

    if not all_X:
        raise RuntimeError(f"No valid CSV files found in {raw_dir}")

    X_combined = np.vstack(all_X).astype(np.float32)
    y_combined = np.concatenate(all_y).astype(np.int64)

    # Shuffle
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X_combined))
    X_combined = X_combined[perm]
    y_combined = y_combined[perm]

    unique, counts = np.unique(y_combined, return_counts=True)
    print(f"\n[Partitioner] Combined pool: {len(X_combined):,} samples, {len(unique)} classes")
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls} ({LABEL_NAMES[cls]}): {cnt:,} ({100*cnt/len(y_combined):.1f}%)")

    return X_combined, y_combined


# ---------------------------------------------------------------------------
# Step 2: Extract globally balanced test set
# ---------------------------------------------------------------------------

def extract_global_test_set(
    X: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a globally balanced test set before client partitioning.

    For each class, takes test_fraction of that class's samples.
    This ensures uniform class distribution in the test set regardless
    of how imbalanced the overall dataset is.

    Returns
    -------
    X_train, y_train : remaining 80% for client partitioning
    X_test,  y_test  : balanced 20% held out globally
    """
    print(f"\n[Partitioner] Extracting global test set ({test_fraction*100:.0f}% per class)...")
    rng = np.random.default_rng(seed)

    test_indices  = []
    train_indices = []

    for cls in range(NUM_CLASSES):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) == 0:
            print(f"  [Warning] Class {cls} has no samples — skipping.")
            continue
        rng.shuffle(cls_idx)
        n_test = max(1, int(len(cls_idx) * test_fraction))
        test_indices.extend(cls_idx[:n_test].tolist())
        train_indices.extend(cls_idx[n_test:].tolist())

    test_indices  = np.array(test_indices,  dtype=np.int64)
    train_indices = np.array(train_indices, dtype=np.int64)

    X_test,  y_test  = X[test_indices],  y[test_indices]
    X_train, y_train = X[train_indices], y[train_indices]

    unique, counts = np.unique(y_test, return_counts=True)
    print(f"  Global test set: {len(X_test):,} samples")
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls} ({LABEL_NAMES[cls]}): {cnt:,} ({100*cnt/len(y_test):.1f}%)")

    print(f"  Remaining for client partitioning: {len(X_train):,} samples")
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Step 3: Dirichlet partitioning
# ---------------------------------------------------------------------------

def dirichlet_partition(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    print(f"\n[Partitioner] Dirichlet(alpha={alpha}) → {n_clients} clients...")
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(y))

    client_indices = [[] for _ in range(n_clients)]

    for cls in range(num_classes):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        if len(cls_indices) == 0:
            continue

        proportions = rng.dirichlet(alpha=alpha * np.ones(n_clients))
        counts = (proportions * len(cls_indices)).astype(int)

        leftover = len(cls_indices) - counts.sum()
        if leftover > 0:
            extra = rng.choice(n_clients, size=leftover, replace=False)
            counts[extra] += 1

        start = 0
        for i in range(n_clients):
            end = start + counts[i]
            client_indices[i].extend(cls_indices[start:end].tolist())
            start = end

    client_data = []
    empty_count = 0

    for i in range(n_clients):
        indices = client_indices[i]
        if len(indices) == 0:
            indices = rng.choice(len(X), size=50, replace=False).tolist()
            empty_count += 1
        idx_arr = np.array(indices, dtype=np.int64)
        rng.shuffle(idx_arr)
        client_data.append((X[idx_arr], y[idx_arr]))

    if empty_count > 0:
        print(f"  [Warning] {empty_count} clients had no samples — assigned random fallback.")

    sizes = [len(cd[0]) for cd in client_data]
    print(f"  Samples per client — min: {min(sizes):,}, max: {max(sizes):,}, mean: {int(np.mean(sizes)):,}")

    return client_data


# ---------------------------------------------------------------------------
# Step 4: Write client CSVs
# ---------------------------------------------------------------------------

def write_client_csvs(
    client_data: List[Tuple[np.ndarray, np.ndarray]],
    out_dir: str,
    n_features: int = 115,
):
    os.makedirs(out_dir, exist_ok=True)
    col_names = [f"feature_{i}" for i in range(n_features)] + ["label"]

    print(f"\n[Partitioner] Writing {len(client_data)} client CSVs to {out_dir}...")

    for i, (X_i, y_i) in enumerate(client_data):
        df = pd.DataFrame(
            np.hstack([X_i, y_i.reshape(-1, 1)]),
            columns=col_names,
        )
        df["label"] = df["label"].astype(int)
        path = os.path.join(out_dir, f"client_{i}.csv")
        df.to_csv(path, index=False)
        if i % 20 == 0 or i == len(client_data) - 1:
            print(f"  Written client_{i}.csv — {len(df):,} samples")

    print(f"[Partitioner] Done.")


# ---------------------------------------------------------------------------
# Step 5: Save global test set
# ---------------------------------------------------------------------------

def save_global_test_set(
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: str,
):
    path = os.path.join(out_dir, "global_test.npz")
    np.savez(path, X=X_test, y=y_test)
    print(f"\n[Partitioner] Global test set saved → {path}")
    print(f"  Shape: X={X_test.shape}, y={y_test.shape}")


# ---------------------------------------------------------------------------
# Non-IID statistics
# ---------------------------------------------------------------------------

def compute_noniid_stats(client_data: List[Tuple], num_classes: int = 11) -> dict:
    entropies = []
    for X_i, y_i in client_data:
        _, counts = np.unique(y_i, return_counts=True)
        probs    = counts / counts.sum()
        entropy  = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

    return {
        "mean_entropy": round(float(np.mean(entropies)), 4),
        "std_entropy":  round(float(np.std(entropies)),  4),
        "min_entropy":  round(float(np.min(entropies)),  4),
        "max_entropy":  round(float(np.max(entropies)),  4),
        "max_possible": round(float(np.log(num_classes)), 4),
    }


def compute_client_distribution(
    client_data: List[Tuple],
    num_classes: int = 11,
) -> dict:
    """Per-client class counts — useful for paper Non-IID analysis table."""
    distribution = {}
    for i, (X_i, y_i) in enumerate(client_data):
        counts = {}
        for cls in range(num_classes):
            counts[str(cls)] = int((y_i == cls).sum())
        distribution[str(i)] = counts
    return distribution


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_partition(raw_dir, out_dir, n_clients, alpha, samples, seed):
    folder = os.path.join(out_dir, f"alpha_{alpha}_N{n_clients}")

    # Check if already done — look for both client CSVs AND global test set
    if os.path.exists(folder):
        existing_csvs = len([
            f for f in os.listdir(folder)
            if f.startswith("client_") and f.endswith(".csv")
        ])
        has_test = os.path.exists(os.path.join(folder, "global_test.npz"))
        if existing_csvs == n_clients and has_test:
            print(f"\n[Partitioner] {folder} already complete — skipping.")
            return folder
        else:
            print(f"\n[Partitioner] {folder} exists but incomplete — regenerating.")

    print(f"\n{'='*60}")
    print(f"  Partitioning: alpha={alpha}, N={n_clients}")
    print(f"{'='*60}")

    # Step 1: Load raw data
    X, y = load_raw_data(raw_dir, samples_per_file=samples)

    # Step 2: Extract globally balanced test set FIRST
    X_train, y_train, X_test, y_test = extract_global_test_set(
        X, y, test_fraction=0.2, seed=seed
    )

    # Step 3: Dirichlet partition the remaining 80%
    client_data = dirichlet_partition(X_train, y_train, n_clients, alpha, seed)

    os.makedirs(folder, exist_ok=True)

    # Step 4: Write client CSVs
    write_client_csvs(client_data, folder, n_features=X.shape[1])

    # Step 5: Save global test set
    save_global_test_set(X_test, y_test, folder)

    # Step 6: Save stats and distribution
    stats = compute_noniid_stats(client_data)
    print(f"\n[Partitioner] Non-IID stats for alpha={alpha}:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    distribution = compute_client_distribution(client_data)

    with open(os.path.join(folder, "partition_stats.json"), "w") as f:
        json.dump({
            "alpha":       alpha,
            "n_clients":   n_clients,
            "seed":        seed,
            "noniid_stats": stats,
            "label_map":   LABEL_MAP,
            "global_test_size": int(len(X_test)),
            "train_pool_size":  int(len(X_train)),
        }, f, indent=2)

    with open(os.path.join(folder, "client_distribution.json"), "w") as f:
        json.dump(distribution, f, indent=2)

    print(f"\n[Partitioner] Partition complete → {folder}/")
    return folder


def parse_args():
    p = argparse.ArgumentParser(description="N-BaIoT Dirichlet Partitioner")
    p.add_argument("--raw-dir",    default="data/raw")
    p.add_argument("--out-dir",    default="data/partitioned")
    p.add_argument("--n-clients",  type=int,   default=200)
    p.add_argument("--alpha",      type=float, default=0.5)
    p.add_argument("--samples",    type=int,   default=50000)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--all-alphas", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.all_alphas:
        for alpha in [0.1, 0.5, 1.0]:
            run_partition(args.raw_dir, args.out_dir,
                          args.n_clients, alpha, args.samples, args.seed)
    else:
        run_partition(args.raw_dir, args.out_dir,
                      args.n_clients, args.alpha, args.samples, args.seed)

    print("\n[Partitioner] All done.")


if __name__ == "__main__":
    main()