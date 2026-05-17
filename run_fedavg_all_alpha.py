"""
run_fedavg_all_alpha.py
=======================

Runs ONLY FedAvg on N-BaIoT using:
    - ALL 115 features (NO chi-square feature selection)
    - All alpha datasets: 0.1, 0.5, 1.0
    - All Byzantine fractions: 0.0, 0.1, 0.2, 0.3
    - Same client pipeline
    - Same ConvMLP model
    - Same Byzantine assignment logic
    - Same evaluation pipeline
    - Same DataManager / NBaiotClient functions

Usage:
    python run_fedavg_all_alpha.py
    python run_fedavg_all_alpha.py --rounds 200
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import ray
import flwr as fl

from datetime import datetime
from typing import Dict, List

from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from torch.utils.data import DataLoader

from src.utils.data_loader import DataManager, NBaiotDataset, CLASS_NAMES
from src.models.conv_mlp import build_model
from src.clients.fl_client import NBaiotClient
from main import assign_byzantine_roles


# ==========================================================================
# CONFIG
# ==========================================================================

DEFAULT_CONFIG = {
    "n_clients":          200,
    "num_classes":        9,
    "seed":               42,
    "num_rounds":         200,
    "fraction_fit":       0.3,
    "local_epochs":       3,
    "local_lr":           0.01,
    "global_lr":          0.1,
    "batch_size":         256,
    "param_bytes":        4,
}

ALPHAS             = [0.1, 0.5, 1.0]
BYZ_FRACTIONS      = [0.0, 0.1, 0.2, 0.3]
RESULTS_DIR        = "results"
INPUT_DIM          = 115    # ALL features — no chi-square


# ==========================================================================
# COMMUNICATION COST
# ==========================================================================

def compute_comm_cost(d: int, config: dict) -> dict:
    """FedAvg always sends full dense gradient."""
    clients_per_round = int(config["n_clients"] * config["fraction_fit"])
    pb                = config["param_bytes"]
    bytes_per_client  = d * pb
    bytes_per_round   = bytes_per_client * clients_per_round
    total_bytes       = bytes_per_round * config["num_rounds"]
    return {
        "kb_per_client_per_round": round(bytes_per_client / 1024, 2),
        "kb_per_round":            round(bytes_per_round  / 1024, 2),
        "total_mb":                round(total_bytes / (1024 * 1024), 2),
    }


# ==========================================================================
# FEDAVG AGGREGATION
# ==========================================================================

def agg_fedavg(g_hats: np.ndarray) -> np.ndarray:
    return g_hats.mean(axis=0)


# ==========================================================================
# STRATEGY
# ==========================================================================

class FedAvgStrategy(fl.server.strategy.Strategy):

    def __init__(self, config, data_manager, byzantine_assignments, input_dim):
        super().__init__()
        self.config                = config
        self.data_manager          = data_manager
        self.byzantine_assignments = byzantine_assignments
        self.input_dim             = input_dim

        self.model          = build_model(input_dim=input_dim,
                                          num_classes=config["num_classes"])
        self.global_weights = self._flatten_model()
        self.d              = len(self.global_weights)

        X_test, y_test      = data_manager.load_global_test_data()
        self.test_dataset   = NBaiotDataset(X_test, y_test)
        self.round_metrics: List[dict] = []

        print(f"[FedAvg] Initialized | d={self.d:,} | "
              f"input_dim={input_dim} | "
              f"byzantine={len(byzantine_assignments)}")

    # ── Flower interface ──────────────────────────────────────────────

    def initialize_parameters(self, client_manager):
        return self._to_params(self.global_weights)

    def configure_fit(self, server_round, parameters, client_manager):
        n_sample = max(10, int(self.config["n_clients"] * self.config["fraction_fit"]))
        clients  = client_manager.sample(num_clients=n_sample, min_num_clients=10)
        fit_ins  = fl.common.FitIns(parameters, {"server_round": server_round})
        return [(c, fit_ins) for c in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        g_list = []
        for proxy, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            g_hat  = np.concatenate([a.ravel() for a in arrays])
            g_list.append(g_hat)
        g_hats              = np.vstack(g_list)
        g_t                 = agg_fedavg(g_hats)
        self.global_weights = self.global_weights + self.config["global_lr"] * g_t
        self._load_weights(self.global_weights)
        return self._to_params(self.global_weights), {}

    def evaluate(self, server_round, parameters):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        loader          = DataLoader(self.test_dataset, batch_size=512, shuffle=False)
        correct, total  = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y    = x.to(device), y.to(device)
                pred    = torch.argmax(self.model(x), dim=1)
                correct += (pred == y).sum().item()
                total   += y.size(0)
        acc = correct / total if total > 0 else 0.0
        print(f"  [FedAvg] Round {server_round:3d} | Acc: {acc:.4f}")
        self.round_metrics.append({"round": server_round, "accuracy": round(float(acc), 4)})
        return float(1 - acc), {"accuracy": float(acc)}

    # ── Helpers ───────────────────────────────────────────────────────

    def _flatten_model(self) -> np.ndarray:
        return np.concatenate([
            p.detach().cpu().numpy().ravel()
            for p in self.model.parameters() if p.requires_grad
        ])

    def _load_weights(self, flat: np.ndarray):
        offset = 0
        with torch.no_grad():
            for p in self.model.parameters():
                if p.requires_grad:
                    size = p.numel()
                    p.copy_(torch.tensor(flat[offset:offset+size].reshape(p.shape),
                                         dtype=torch.float32))
                    offset += size

    def _to_params(self, flat: np.ndarray):
        arrays, offset = [], 0
        for p in self.model.parameters():
            if p.requires_grad:
                size = p.numel()
                arrays.append(flat[offset:offset+size].reshape(p.shape))
                offset += size
        return ndarrays_to_parameters(arrays)


# ==========================================================================
# CLIENT FACTORY
# ==========================================================================

def make_client_fn(data_manager, byzantine_assignments, config, input_dim):
    def client_fn(context):
        cid     = int(context.node_config["partition-id"]) % config["n_clients"]
        X, y    = data_manager.load_client_data(cid)
        dataset = NBaiotDataset(X, y)
        return NBaiotClient(
            client_id      = cid,
            dataset        = dataset,
            input_dim      = input_dim,
            num_classes    = config["num_classes"],
            local_epochs   = config["local_epochs"],
            local_lr       = config["local_lr"],
            batch_size     = config["batch_size"],
            is_byzantine   = (cid in byzantine_assignments),
            attack_type    = byzantine_assignments.get(cid, None),
            top_k_ratio    = 1.0,   # full dense gradient — no sparsification
            residual_decay = 0.9,
        ).to_client()
    return client_fn


# ==========================================================================
# FINAL EVALUATION
# ==========================================================================

def evaluate_final(flat_weights, data_manager, input_dim, config) -> dict:
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
    loader         = DataLoader(NBaiotDataset(X_test, y_test),
                                batch_size=512, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = torch.argmax(model(x), dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    n_classes  = config["num_classes"]

    # Per-class F1
    f1_per_class = f1_score(
        all_labels, all_preds,
        average=None,
        zero_division=0,
        labels=list(range(n_classes)),
    ).tolist()

    # Per-class recall
    recall_per_class = recall_score(
        all_labels, all_preds,
        average=None,
        zero_division=0,
        labels=list(range(n_classes)),
    ).tolist()

    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    return {
        "accuracy":         round(float(accuracy_score(all_labels, all_preds)), 4),
        "f1_macro":         round(float(f1_score(all_labels, all_preds,
                                                  average="macro", zero_division=0)), 4),
        "precision_macro":  round(float(precision_score(all_labels, all_preds,
                                                         average="macro", zero_division=0)), 4),
        "recall_macro":     round(float(recall_score(all_labels, all_preds,
                                                      average="macro", zero_division=0)), 4),
        "f1_per_class":     [round(v, 4) for v in f1_per_class],
        "recall_per_class": [round(v, 4) for v in recall_per_class],
        "classification_report": report,
    }


# ==========================================================================
# RUN ONE SCENARIO (one alpha + one byzantine fraction)
# ==========================================================================

def run_scenario(alpha: float, byz_fraction: float, config: dict) -> dict:

    scenario = "No Attack" if byz_fraction == 0.0 else f"{int(byz_fraction*100)}% Byzantine"
    print(f"\n{'='*65}")
    print(f"  Alpha={alpha}  |  {scenario}")
    print(f"{'='*65}")

    data_dir = f"data/partitioned/alpha_{alpha}_N200"
    dm       = DataManager(data_dir=data_dir)

    # No feature selection — all 115 features
    input_dim = INPUT_DIM

    # Byzantine assignment
    byzantine_assignments = assign_byzantine_roles(
        config["n_clients"], byz_fraction, config["seed"]
    ) if byz_fraction > 0.0 else {}

    strategy  = FedAvgStrategy(
        config=config,
        data_manager=dm,
        byzantine_assignments=byzantine_assignments,
        input_dim=input_dim,
    )
    client_fn = make_client_fn(dm, byzantine_assignments, config, input_dim)

    t_start = time.time()
    fl.simulation.start_simulation(
        client_fn       = client_fn,
        num_clients     = config["n_clients"],
        config          = fl.server.ServerConfig(num_rounds=config["num_rounds"]),
        strategy        = strategy,
        client_resources= {"num_cpus": 2, "num_gpus": 0.0},
    )
    total_time = time.time() - t_start

    final_metrics  = evaluate_final(strategy.global_weights, dm, input_dim, config)
    comm           = compute_comm_cost(strategy.d, config)
    time_per_round = round(total_time / config["num_rounds"], 2)

    # Best accuracy over all rounds
    per_round_acc = [m["accuracy"] for m in strategy.round_metrics]
    best_acc      = max(per_round_acc) if per_round_acc else final_metrics["accuracy"]
    best_round    = per_round_acc.index(best_acc) + 1 if per_round_acc else config["num_rounds"]

    result = {
        "alpha":              alpha,
        "byzantine_fraction": byz_fraction,
        "scenario":           scenario,
        "accuracy":           final_metrics["accuracy"],
        "f1_macro":           final_metrics["f1_macro"],
        "precision_macro":    final_metrics["precision_macro"],
        "recall_macro":       final_metrics["recall_macro"],
        "f1_per_class":       final_metrics["f1_per_class"],
        "recall_per_class":   final_metrics["recall_per_class"],
        "best_accuracy":      round(best_acc, 4),
        "best_round":         best_round,
        "comm_kb_per_round":  comm["kb_per_round"],
        "comm_total_mb":      comm["total_mb"],
        "total_time_s":       round(total_time, 2),
        "time_per_round_s":   time_per_round,
        "per_round_accuracy": per_round_acc,
        "classification_report": final_metrics["classification_report"],
    }

    print(f"\n  [alpha={alpha} | {scenario}] DONE")
    print(f"  Accuracy  : {result['accuracy']:.4f}  (best={best_acc:.4f} @ round {best_round})")
    print(f"  F1 (macro): {result['f1_macro']:.4f}")
    print(f"  Precision : {result['precision_macro']:.4f}")
    print(f"  Recall    : {result['recall_macro']:.4f}")
    print(f"  KB/round  : {comm['kb_per_round']:.1f}")
    print(f"  Time      : {total_time:.1f}s  ({time_per_round:.1f}s/round)")

    return result


# ==========================================================================
# PRINT FINAL TABLE
# ==========================================================================

def print_final_table(all_results: List[dict]):
    sep  = "=" * 110
    sep2 = "-" * 110

    lines = []
    lines.append(f"\n\n{sep}")
    lines.append("  FEDAVG — ALL ALPHAS × ALL BYZANTINE FRACTIONS  (115 features, no chi-square)")
    lines.append(sep)
    lines.append(
        f"  {'Alpha':>6}  {'Scenario':<22}  {'Accuracy':>9}  {'Best':>6}  "
        f"{'F1':>7}  {'Precision':>10}  {'Recall':>8}  "
        f"{'KB/Round':>10}  {'Time/Rnd':>9}"
    )
    lines.append(sep2)

    for r in all_results:
        lines.append(
            f"  {r['alpha']:>6}  {r['scenario']:<22}  "
            f"{r['accuracy']:>9.4f}  {r['best_accuracy']:>6.4f}  "
            f"{r['f1_macro']:>7.4f}  {r['precision_macro']:>10.4f}  "
            f"{r['recall_macro']:>8.4f}  "
            f"{r['comm_kb_per_round']:>10.1f}  {r['time_per_round_s']:>9.1f}"
        )

    lines.append(sep)

    # Per-class F1 table
    lines.append(f"\n  Per-class F1 Score:")
    header = f"  {'Alpha':>6}  {'Scenario':<22}  " + \
             "  ".join([f"{n[:8]:>8}" for n in CLASS_NAMES])
    lines.append(header)
    lines.append(sep2)
    for r in all_results:
        cells = "  ".join([f"{v:>8.4f}" for v in r["f1_per_class"]])
        lines.append(f"  {r['alpha']:>6}  {r['scenario']:<22}  {cells}")
    lines.append(sep)

    # Per-class Recall table
    lines.append(f"\n  Per-class Recall:")
    lines.append(header.replace("F1", "Recall"))
    lines.append(sep2)
    for r in all_results:
        cells = "  ".join([f"{v:>8.4f}" for v in r["recall_per_class"]])
        lines.append(f"  {r['alpha']:>6}  {r['scenario']:<22}  {cells}")
    lines.append(sep)

    output = "\n".join(lines)
    print(output)

    # Save table to file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table_path = os.path.join(RESULTS_DIR, "fedavg_all_alpha_final_table.txt")
    with open(table_path, "w") as f:
        f.write(output)
    print(f"\n  Table saved → {table_path}")


# ==========================================================================
# MAIN
# ==========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=DEFAULT_CONFIG["num_rounds"])
    p.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    p.add_argument("--byz-fractions", nargs="+", type=float, default=BYZ_FRACTIONS)
    return p.parse_args()


def main():
    args   = parse_args()
    config = {**DEFAULT_CONFIG, "num_rounds": args.rounds}

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print("  FedAvg — All Alphas × All Byzantine Fractions")
    print("=" * 65)
    print(f"  Features     : ALL 115 (no chi-square)")
    print(f"  Alphas       : {args.alphas}")
    print(f"  Byz fractions: {args.byz_fractions}")
    print(f"  Rounds       : {config['num_rounds']}")
    print(f"  Clients      : {config['n_clients']}")
    print("=" * 65)

    all_results = []

    for alpha in args.alphas:
        for byz_f in args.byz_fractions:

            # Shutdown and reinit Ray between each scenario
            if ray.is_initialized():
                ray.shutdown()
            ray.init(
                num_cpus=4,
                object_store_memory=512 * 1024 * 1024,
            )

            try:
                result = run_scenario(alpha, byz_f, config)
                all_results.append(result)

            except Exception as e:
                import traceback
                print(f"\n[ERROR] alpha={alpha}, byz={byz_f}: {e}")
                traceback.print_exc()
                all_results.append({
                    "alpha": alpha,
                    "byzantine_fraction": byz_f,
                    "scenario": f"{int(byz_f*100)}% Byzantine",
                    "error": str(e),
                })

    # Print final table
    valid = [r for r in all_results if "accuracy" in r]
    if valid:
        print_final_table(valid)

    # Save full results JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(RESULTS_DIR, f"fedavg_all_alpha_{timestamp}.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results saved → {save_path}")

    print("\n[Main] All experiments complete.")


if __name__ == "__main__":
    main()