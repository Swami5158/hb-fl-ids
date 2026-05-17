"""
main.py — HB-FL-IDS Full Experiment Runner
==========================================
Full system with Byzantine defense, communication efficiency,
and server-side evaluation on globally balanced test set.

Prerequisites:
    python partition_data.py --all-alphas --n-clients 200

Single experiment:
    python main.py --alpha 0.5 --f 0.2 --rounds 100

Clean baseline (no Byzantine):
    python main.py --alpha 0.5 --f 0.0 --no-attack --rounds 100

Full 9-experiment matrix:
    python main.py --run-all --rounds 100
"""  

import os
import json
import argparse
import numpy as np
import torch
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
)
from torch.utils.data import DataLoader


from src.utils.data_loader import DataManager, NBaiotDataset
from src.utils.feature_selection import run_federated_feature_selection
from src.models.conv_mlp import build_model
from src.server.fl_server import run_simulation


# ---------------------------------------------------------------------------
# Configuration defaults — tuned from baseline experiments
# ---------------------------------------------------------------------------

DEFAULTS = {
    "n_clients":          200,
    "num_rounds":         100,
    "fraction_fit":       0.2,        # 40 clients/round
    "alpha_dir":          0.5,
    "byzantine_fraction": 0.2,
    "global_lr":          0.1,
    "local_lr":           0.01,
    "local_epochs":       3,
    "fsel":               70,
    "num_classes":        9,
    "batch_size":         256,
    "seed":               42,
    "top_k_ratio":        0.3,        # Algorithm 2: rho=0.1
    "residual_decay":     0.9,        # Algorithm 2: xi=0.9
    "beta":               0.9,        # EMA reputation decay
    "gamma_0":            0.1,        # initial trimming ratio
    "lambda_adapt":       0.1,        # sigma^2 adaptation rate
    "gamma_min":          0.05,       # minimum trimming ratio
    "lambda_m":           1.5,        # MZ-score magnitude threshold
    "lambda_d":           1.5,        # MZ-score PDP threshold
}

# theta_min per alpha — more Non-IID -> lower threshold
# (honest clients naturally have more gradient diversity)
THETA_MIN_MAP = {0.1: 0.0, 0.5: 0.1, 1.0: 0.2}

ALPHA_VALUES  = [0.1, 0.5, 1.0]
F_VALUES      = [0.0, 0.1, 0.2, 0.3]

ATTACK_TYPES  = [
    "label_flip", "gradient_poison", "scaling",
    "random_noise", "backdoor", "min_max",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="HB-FL-IDS Experiment Runner")
    p.add_argument("--data-dir",       default="data/partitioned")
    p.add_argument("--results-dir",    default="results")
    p.add_argument("--n-clients",      type=int,   default=DEFAULTS["n_clients"])
    p.add_argument("--rounds",         type=int,   default=DEFAULTS["num_rounds"])
    p.add_argument("--alpha",          type=float, default=DEFAULTS["alpha_dir"])
    p.add_argument("--f",              type=float, default=DEFAULTS["byzantine_fraction"])
    p.add_argument("--global-lr",      type=float, default=DEFAULTS["global_lr"])
    p.add_argument("--local-lr",       type=float, default=DEFAULTS["local_lr"])
    p.add_argument("--epochs",         type=int,   default=DEFAULTS["local_epochs"])
    p.add_argument("--fsel",           type=int,   default=DEFAULTS["fsel"])
    p.add_argument("--seed",           type=int,   default=DEFAULTS["seed"])
    p.add_argument("--fraction-fit",   type=float, default=DEFAULTS["fraction_fit"])
    p.add_argument("--top-k-ratio",    type=float, default=DEFAULTS["top_k_ratio"])
    p.add_argument("--residual-decay", type=float, default=DEFAULTS["residual_decay"])
    p.add_argument("--no-attack",      action="store_true")
    p.add_argument("--skip-fs",        action="store_true")
    p.add_argument("--run-all",        action="store_true",
                   help="Run full experiment matrix")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Main] Seed set to {seed}")


def get_data_dir(base_dir: str, alpha: float, n_clients: int) -> str:
    folder = os.path.join(base_dir, f"alpha_{alpha}_N{n_clients}")
    if not os.path.exists(folder):
        raise RuntimeError(
            f"Data not found at {folder}.\n"
            f"Run: python partition_data.py --n-clients {n_clients} --alpha {alpha}"
        )
    if not os.path.exists(os.path.join(folder, "global_test.npz")):
        raise RuntimeError(
            f"global_test.npz not found in {folder}.\n"
            f"Re-run partition_data.py to regenerate with global test set."
        )
    return folder


def setup_results_dir(results_dir: str, config: dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = (
        f"run_{timestamp}"
        f"_N{config['n_clients']}"
        f"_alpha{str(config['alpha_dir']).replace('.','')}"
        f"_f{int(config['byzantine_fraction']*100)}"
        f"_T{config['num_rounds']}"
    )
    run_dir = os.path.join(results_dir, folder)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Main] Results → {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Byzantine role assignment — Algorithm 5 Line 3
# ---------------------------------------------------------------------------

def assign_byzantine_roles(
    n_clients: int,
    byzantine_fraction: float,
    seed: int = 42,
) -> Dict[int, str]:
    """
    Assign attack types to floor(f*N) clients.
    Uses sorted deterministic selection so that higher byzantine_fraction
    always includes all clients from lower fractions.
    e.g. 20% Byzantine set always contains the full 10% Byzantine set.
    """
    rng = np.random.default_rng(seed)
    n_byzantine = int(np.floor(byzantine_fraction * n_clients))

    if n_byzantine == 0:
        return {}

    # Generate a fixed random permutation of all client IDs
    # First n_byzantine clients in this permutation are always selected
    # This ensures 10% set ⊆ 20% set ⊆ 30% set
    all_ids = rng.permutation(n_clients).tolist()
    byzantine_ids = all_ids[:n_byzantine]

    assignments = {}
    for idx, cid in enumerate(byzantine_ids):
        assignments[cid] = ATTACK_TYPES[idx % len(ATTACK_TYPES)]

    dist = Counter(assignments.values())
    print(f"[Main] Byzantine: {n_byzantine}/{n_clients} clients")
    print(f"[Main] Attack distribution: {dict(dist)}")
    return assignments

# ---------------------------------------------------------------------------
# Final evaluation on global test set
# ---------------------------------------------------------------------------

def evaluate_global_model(
    final_weights: np.ndarray,
    data_manager: DataManager,
    input_dim: int,
    num_classes: int = 9,
    batch_size: int = 256,
) -> tuple:
    print("\n[Eval] Final evaluation on global balanced test set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(input_dim=input_dim, num_classes=num_classes)
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                size  = p.numel()
                chunk = final_weights[offset:offset + size]
                p.copy_(torch.tensor(chunk.reshape(p.shape), dtype=torch.float32))
                offset += size

    model = model.to(device)
    model.eval()

    X_test, y_test = data_manager.load_global_test_data()
    dataset = NBaiotDataset(X_test, y_test)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            logits  = model(batch_X)
            probs   = torch.softmax(logits, dim=1).cpu().numpy()
            preds   = np.argmax(probs, axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.numpy().tolist())
            all_probs.extend(probs.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    accuracy  = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall    = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        auc = float("nan")

    metrics = {
        "accuracy":  round(float(accuracy),  4),
        "f1_macro":  round(float(f1),        4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall),    4),
        "auc_roc":   round(float(auc), 4) if not np.isnan(auc) else None,
        "n_samples": len(all_labels),
    }

    print(f"\n{'='*55}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    if not np.isnan(auc):
        print(f"  AUC-ROC   : {auc:.4f}")
    else:
        print(f"  AUC-ROC   : N/A")
    print(f"  Samples   : {len(all_labels):,} (global balanced test set)")
    print(f"{'='*55}\n")

    report = classification_report(
        all_labels, all_preds,
        target_names=[str(i) for i in range(num_classes)],
        zero_division=0,
    )
    print(report)
    return metrics, report


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(run_dir, final_metrics, report, round_metrics, selected_features):
    with open(os.path.join(run_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    with open(os.path.join(run_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(run_dir, "round_metrics.json"), "w") as f:
        json.dump(round_metrics, f, indent=2, default=str)
    np.save(os.path.join(run_dir, "selected_features.npy"), selected_features)
    print(f"[Main] Results saved to {run_dir}/")


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_experiment(config: dict, base_data_dir: str) -> dict:
    run_dir  = setup_results_dir(config["results_dir"], config)
    data_dir = get_data_dir(base_data_dir, config["alpha_dir"], config["n_clients"])
    print(f"[Main] Data dir: {data_dir}")

    dm         = DataManager(data_dir=data_dir)
    client_ids = dm.get_client_ids()
    print(f"[Main] Found {len(client_ids)} client files.")

    # Algorithm 1 — Federated Chi-square on ALL clients
    if config.get("skip_fs", False):
        print("[Main] Skipping Algorithm 1 — using all 115 features.")
        selected_features = np.arange(115)
        input_dim = 115
        dm.apply_feature_selection(selected_features)
    else:
        print(f"[Main] Algorithm 1: Chi-square on all {len(client_ids)} clients...")
        selected_features = run_federated_feature_selection(
            data_manager=dm,
            client_ids=client_ids,
            fsel=config["fsel"],
            num_classes=config["num_classes"],
        )
        # run_federated_feature_selection calls dm.apply_feature_selection internally
        input_dim = len(selected_features)

    print(f"[Main] {input_dim} features active.")

    # Byzantine assignment
    byzantine_assignments = assign_byzantine_roles(
        n_clients=config["n_clients"],
        byzantine_fraction=config["byzantine_fraction"],
        seed=config["seed"],
    ) if not config.get("no_attack", False) else {}

    # FL simulation
    history, round_metrics, final_weights = run_simulation(
        data_manager=dm,
        n_clients=config["n_clients"],
        byzantine_assignments=byzantine_assignments,
        num_rounds=config["num_rounds"],
        fraction_fit=config["fraction_fit"],
        global_lr=config["global_lr"],
        byzantine_fraction=config["byzantine_fraction"],
        theta_min=THETA_MIN_MAP.get(config["alpha_dir"], 0.0),
        input_dim=input_dim,
        num_classes=config["num_classes"],
        local_epochs=config["local_epochs"],
        local_lr=config["local_lr"],
        batch_size=config["batch_size"],
        top_k_ratio=config["top_k_ratio"],
        residual_decay=config["residual_decay"],
        beta=config["beta"],
        gamma_0=config["gamma_0"],
        lambda_adapt=config["lambda_adapt"],
        gamma_min=config["gamma_min"],
        lambda_m=config["lambda_m"],
        lambda_d=config["lambda_d"],
    )

    final_metrics, report = evaluate_global_model(
        final_weights=final_weights,
        data_manager=dm,
        input_dim=input_dim,
        num_classes=config["num_classes"],
        batch_size=config["batch_size"],
    )
    final_metrics["config"] = config

    save_results(run_dir, final_metrics, report, round_metrics, selected_features)

    print("\n[Main] Round-by-round global accuracy:")
    for m in round_metrics:
        acc = m.get("global_accuracy", None)
        rnd = m.get("server_round", "?")
        if isinstance(acc, float):
            print(f"  Round {rnd:3d}: {acc:.4f}")

    return final_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    config = {**DEFAULTS}
    config.update({
        "results_dir":        args.results_dir,
        "n_clients":          args.n_clients,
        "num_rounds":         args.rounds,
        "alpha_dir":          args.alpha,
        "byzantine_fraction": 0.0 if args.no_attack else args.f,
        "global_lr":          args.global_lr,
        "local_lr":           args.local_lr,
        "local_epochs":       args.epochs,
        "fsel":               args.fsel,
        "seed":               args.seed,
        "no_attack":          args.no_attack,
        "skip_fs":            args.skip_fs,
        "fraction_fit":       args.fraction_fit,
        "top_k_ratio":        args.top_k_ratio,
        "residual_decay":     args.residual_decay,
    })

    print("\n" + "="*60)
    print("  HB-FL-IDS: Hybrid Byzantine-Resilient FL-IDS")
    print("="*60)
    print(f"  Clients      : {config['n_clients']}")
    print(f"  Rounds       : {config['num_rounds']}")
    print(f"  Per-round    : {int(config['n_clients'] * config['fraction_fit'])} clients")
    print(f"  Alpha        : {config['alpha_dir']}")
    print(f"  Byzantine f  : {config['byzantine_fraction']}")
    print(f"  Attack mode  : {'DISABLED' if args.no_attack else 'ENABLED'}")
    print(f"  rho (top-K)  : {config['top_k_ratio']}")
    print(f"  xi (residual): {config['residual_decay']}")
    print(f"  global_lr    : {config['global_lr']}")
    print(f"  local_lr     : {config['local_lr']}")
    print(f"  Evaluation   : server-side on global balanced test set")
    print("="*60 + "\n")

    os.makedirs(args.results_dir, exist_ok=True)

    if args.run_all:
        print(f"[Main] Full matrix: "
              f"{len(ALPHA_VALUES)} alpha × {len(F_VALUES)} f = "
              f"{len(ALPHA_VALUES) * len(F_VALUES)} experiments\n")

        summary = []
        for alpha in ALPHA_VALUES:
            for f_val in F_VALUES:
                print(f"\n{'='*60}")
                print(f"  EXPERIMENT: alpha={alpha}, f={f_val}")
                print(f"{'='*60}")

                # Reset RNG so each experiment is reproducible standalone
                set_seed(config["seed"])

                exp_config = {
                    **config,
                    "alpha_dir":          alpha,
                    "byzantine_fraction": f_val,
                    "no_attack":          (f_val == 0.0),
                }
                try:
                    metrics = run_experiment(exp_config, args.data_dir)
                    summary.append({
                        "alpha":    alpha,
                        "f":        f_val,
                        "accuracy": metrics["accuracy"],
                        "f1_macro": metrics["f1_macro"],
                        "auc_roc":  metrics.get("auc_roc"),
                    })
                    print(f"  Done: acc={metrics['accuracy']:.4f}, "
                          f"f1={metrics['f1_macro']:.4f}")
                except Exception as e:
                    print(f"  [Error] alpha={alpha}, f={f_val}: {e}")
                    summary.append({
                        "alpha": alpha, "f": f_val,
                        "accuracy": None, "f1_macro": None,
                        "error": str(e),
                    })

        summary_path = os.path.join(args.results_dir, "experiment_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*60)
        print(f"{'Alpha':>8} {'f':>6} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
        print("-"*60)
        for row in summary:
            acc = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "ERROR"
            f1  = f"{row['f1_macro']:.4f}"  if row["f1_macro"] is not None else "ERROR"
            auc = f"{row['auc_roc']:.4f}"   if row.get("auc_roc") is not None else "N/A"
            print(f"{row['alpha']:>8} {row['f']:>6} {acc:>10} {f1:>10} {auc:>10}")
        print("="*60)
        print(f"\n[Main] Summary saved to {summary_path}")

    else:
        run_experiment(config, args.data_dir)

    print("\n[Main] All experiments complete.")


if __name__ == "__main__":
    main()
