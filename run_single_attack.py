"""
run_single_attack.py
====================
Runs ONE method with ONE attack type.
alpha=0.5, 30% Byzantine, 100 rounds.

Usage:
    python run_single_attack.py --method fedavg --attack gradient_poison
    python run_single_attack.py --method fltrust --attack scaling
    python run_single_attack.py --method hb_fl_ids --attack min_max
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import ray
import flwr as fl

from datetime import datetime
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.utils.data_loader import DataManager, NBaiotDataset, CLASS_NAMES
from src.utils.feature_selection import run_federated_feature_selection
from src.models.conv_mlp import build_model
from src.clients.fl_client import NBaiotClient
from main import assign_byzantine_roles


VALID_METHODS = ["fedavg", "fltrust", "hb_fl_ids"]
VALID_ATTACKS = ["label_flip", "gradient_poison", "scaling",
                 "random_noise", "backdoor", "min_max"]

CONFIG = {
    "data_dir":           "data/partitioned/alpha_0.5_N200",
    "n_clients":          200,
    "num_classes":        9,
    "fsel":               70,
    "seed":               42,
    "num_rounds":         100,
    "fraction_fit":       0.3,
    "local_epochs":       3,
    "local_lr":           0.01,
    "global_lr":          0.1,
    "batch_size":         256,
    "byzantine_fraction": 0.3,
    "top_k_ratio":        0.4,
    "residual_decay":     0.9,
    "beta":               0.9,
    "gamma_0":            0.05,
    "lambda_adapt":       0.1,
    "gamma_min":          0.02,
    "lambda_m":           1.5,
    "lambda_d":           1.5,
    "root_dataset_size":  100,
    "param_bytes":        4,
    "index_bytes":        4,
}


def assign_single_attack(n_clients, fraction, attack_type, seed=42):
    """All Byzantine clients use the SAME single attack type."""
    rng = np.random.default_rng(seed)
    n_byz = int(np.floor(fraction * n_clients))
    if n_byz == 0:
        return {}
    all_ids = rng.permutation(n_clients).tolist()
    byzantine_ids = all_ids[:n_byz]
    assignments = {cid: attack_type for cid in byzantine_ids}
    print(f"[Byzantine] {n_byz}/{n_clients} clients all using: {attack_type}")
    return assignments


def compute_comm_cost(d, config, method):
    clients_per_round = int(config["n_clients"] * config["fraction_fit"])
    pb = config["param_bytes"]
    ib = config["index_bytes"]
    if method == "hb_fl_ids":
        k = int(config["top_k_ratio"] * d)
        bpc = k * pb + k * ib
    else:
        bpc = d * pb
    bpr = bpc * clients_per_round
    return {
        "kb_per_round":           round(bpr / 1024, 2),
        "reduction_vs_dense_pct": round((1 - bpc / (d * pb)) * 100, 1)
                                   if method == "hb_fl_ids" else 0.0,
    }


def evaluate_final(flat_weights, data_manager, input_dim, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(input_dim=input_dim, num_classes=config["num_classes"])
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                size = p.numel()
                p.copy_(torch.tensor(
                    flat_weights[offset:offset+size].reshape(p.shape),
                    dtype=torch.float32))
                offset += size
    model.to(device)
    model.eval()

    X_test, y_test = data_manager.load_global_test_data()
    loader = DataLoader(NBaiotDataset(X_test, y_test), batch_size=512, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = torch.argmax(model(x), dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    f1_per_class = f1_score(all_labels, all_preds, average=None,
                            zero_division=0, labels=list(range(config["num_classes"]))).tolist()

    return {
        "accuracy":         round(float(accuracy_score(all_labels, all_preds)), 4),
        "f1_macro":         round(float(f1_score(all_labels, all_preds,
                                                  average="macro", zero_division=0)), 4),
        "precision_macro":  round(float(precision_score(all_labels, all_preds,
                                                         average="macro", zero_division=0)), 4),
        "recall_macro":     round(float(recall_score(all_labels, all_preds,
                                                      average="macro", zero_division=0)), 4),
        "f1_per_class":     [round(v, 4) for v in f1_per_class],
    }


def run_single(method, attack_type, config):
    print(f"\n{'='*60}")
    print(f"  Method: {method.upper()}  |  Attack: {attack_type}")
    print(f"  Alpha=0.5  |  Byzantine=30%  |  Rounds={config['num_rounds']}")
    print(f"{'='*60}")

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=4, object_store_memory=512*1024*1024)

    dm = DataManager(data_dir=config["data_dir"])

    # Feature selection
    client_ids = dm.get_client_ids()
    selected   = run_federated_feature_selection(
        data_manager=dm,
        client_ids=client_ids,
        fsel=config["fsel"],
        num_classes=config["num_classes"],
    )
    input_dim = len(selected)

    # Single attack Byzantine assignment
    byzantine_assignments = assign_single_attack(
        config["n_clients"],
        config["byzantine_fraction"],
        attack_type,
        config["seed"],
    )

    t_start = time.time()

    if method == "hb_fl_ids":
        from src.server.fl_server import run_simulation
        history, round_metrics, final_weights = run_simulation(
            data_manager         = dm,
            n_clients            = config["n_clients"],
            byzantine_assignments= byzantine_assignments,
            num_rounds           = config["num_rounds"],
            fraction_fit         = config["fraction_fit"],
            global_lr            = config["global_lr"],
            byzantine_fraction   = config["byzantine_fraction"],
            theta_min            = 0.1,
            input_dim            = input_dim,
            num_classes          = config["num_classes"],
            local_epochs         = config["local_epochs"],
            local_lr             = config["local_lr"],
            batch_size           = config["batch_size"],
            top_k_ratio          = config["top_k_ratio"],
            residual_decay       = config["residual_decay"],
            beta                 = config["beta"],
            gamma_0              = config["gamma_0"],
            lambda_adapt         = config["lambda_adapt"],
            gamma_min            = config["gamma_min"],
            lambda_m             = config["lambda_m"],
            lambda_d             = config["lambda_d"],
        )
        per_round = [m.get("global_accuracy") for m in round_metrics]

    else:
        from run_all_experiments import BaselineStrategy, make_client_fn

        root_dataset = None
        if method == "fltrust":
            X_test, y_test = dm.load_global_test_data()
            rng = np.random.default_rng(config["seed"])
            n_per_class = max(1, config["root_dataset_size"] // config["num_classes"])
            root_indices = []
            for cls in range(config["num_classes"]):
                cls_idx = np.where(y_test == cls)[0]
                n = min(n_per_class, len(cls_idx))
                chosen = rng.choice(cls_idx, size=n, replace=False)
                root_indices.extend(chosen.tolist())
            root_dataset = NBaiotDataset(X_test[np.array(root_indices)],
                                          y_test[np.array(root_indices)])

        strategy = BaselineStrategy(
            method=method,
            config=config,
            data_manager=dm,
            byzantine_assignments=byzantine_assignments,
            input_dim=input_dim,
            root_dataset=root_dataset,
        )
        client_fn = make_client_fn(dm, byzantine_assignments, config, input_dim, False)

        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=config["n_clients"],
            config=fl.server.ServerConfig(num_rounds=config["num_rounds"]),
            strategy=strategy,
            client_resources={"num_cpus": 2, "num_gpus": 0.0},
        )
        final_weights = strategy.global_weights
        per_round     = [m["accuracy"] for m in strategy.round_metrics]

    total_time = time.time() - t_start

    tmp_model = build_model(input_dim=input_dim, num_classes=config["num_classes"])
    d = tmp_model.get_gradient_dim()

    final_metrics = evaluate_final(final_weights, dm, input_dim, config)
    comm          = compute_comm_cost(d, config, method)

    per_round_clean = [v for v in per_round if v is not None]
    best_acc   = max(per_round_clean) if per_round_clean else final_metrics["accuracy"]
    best_round = per_round_clean.index(best_acc) + 1 if per_round_clean else 0

    result = {
        "method":             method,
        "attack_type":        attack_type,
        "accuracy":           final_metrics["accuracy"],
        "f1_macro":           final_metrics["f1_macro"],
        "precision_macro":    final_metrics["precision_macro"],
        "recall_macro":       final_metrics["recall_macro"],
        "f1_per_class":       final_metrics["f1_per_class"],
        "best_accuracy":      round(best_acc, 4),
        "best_round":         best_round,
        "kb_per_round":       comm["kb_per_round"],
        "comm_reduction_pct": comm["reduction_vs_dense_pct"],
        "total_time_s":       round(total_time, 2),
        "time_per_round_s":   round(total_time / config["num_rounds"], 2),
        "per_round_accuracy": per_round_clean,
    }

    print(f"\n  DONE: {method.upper()} | {attack_type}")
    print(f"  Accuracy  : {result['accuracy']:.4f}  (best={best_acc:.4f} @ round {best_round})")
    print(f"  F1 (macro): {result['f1_macro']:.4f}")
    print(f"  KB/round  : {comm['kb_per_round']:.1f}  "
          f"({comm['reduction_vs_dense_pct']}% reduction)")
    print(f"  Time      : {total_time:.1f}s  ({result['time_per_round_s']:.1f}s/round)")

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=VALID_METHODS)
    p.add_argument("--attack", required=True, choices=VALID_ATTACKS)
    p.add_argument("--rounds", type=int, default=CONFIG["num_rounds"])
    args = p.parse_args()

    CONFIG["num_rounds"] = args.rounds

    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    os.makedirs("results", exist_ok=True)

    result = run_single(args.method, args.attack, CONFIG)

    # Save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/single_{args.method}_{args.attack}_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n  Result saved → {path}")
    print(f"\n{'='*60}")
    print(f"  FINAL: {args.method.upper()} | {args.attack}")
    print(f"  Accuracy : {result['accuracy']:.4f}")
    print(f"  F1       : {result['f1_macro']:.4f}")
    print(f"  KB/round : {result['kb_per_round']:.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
