"""
run_fedavg_single_attacks_final.py
==================================

FINAL research-grade FedAvg benchmark for SINGLE Byzantine attacks.

Features:
    - ONLY FedAvg
    - ALL 115 features (NO chi-square)
    - Single attack scenarios
    - All alphas: 0.1, 0.5, 1.0
    - All Byzantine fractions: 0.1, 0.2, 0.3
    - Full IDS evaluation metrics
    - Communication analysis
    - Convergence analysis
    - Per-class metrics
    - Best-round tracking

Attack Types:
    1. label_flip
    2. gradient_poison
    3. scaling
    4. random_noise
    5. backdoor
    6. min_max

Usage:
    python run_fedavg_single_attacks_final.py

Optional:
    python run_fedavg_single_attacks_final.py --rounds 200
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
from typing import List

from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from torch.utils.data import DataLoader

from src.utils.data_loader import (
    DataManager,
    NBaiotDataset,
    CLASS_NAMES,
)

from src.models.conv_mlp import build_model
from src.clients.fl_client import NBaiotClient


# ==========================================================================
# CONFIG
# ==========================================================================

DEFAULT_CONFIG = {

    "n_clients":          200,
    "num_classes":        9,
    "seed":               42,

    "num_rounds":         100,

    "fraction_fit":       0.3,

    "local_epochs":       3,

    "local_lr":           0.01,

    "global_lr":          0.1,

    "batch_size":         256,

    "param_bytes":        4,
}

ALPHAS = [0.1, 0.5, 1.0]

BYZ_FRACTIONS = [0.1, 0.2, 0.3]

ATTACK_TYPES = [
    "label_flip",
    "gradient_poison",
    "scaling",
    "random_noise",
    "backdoor",
    "min_max",
]

INPUT_DIM = 115

RESULTS_DIR = "results"


# ==========================================================================
# BYZANTINE ASSIGNMENT
# ==========================================================================

def assign_single_attack_roles(
    n_clients: int,
    byz_fraction: float,
    attack_type: str,
    seed: int,
):

    rng = np.random.default_rng(seed)

    n_byz = int(n_clients * byz_fraction)

    byz_clients = rng.choice(
        np.arange(n_clients),
        size=n_byz,
        replace=False,
    )

    assignments = {
        int(cid): attack_type
        for cid in byz_clients
    }

    print(
        f"[Single Attack] "
        f"{n_byz}/{n_clients} Byzantine | "
        f"attack={attack_type}"
    )

    return assignments


# ==========================================================================
# COMMUNICATION COST
# ==========================================================================

def compute_comm_cost(d: int, config: dict):

    clients_per_round = int(
        config["n_clients"] *
        config["fraction_fit"]
    )

    bytes_per_client = (
        d * config["param_bytes"]
    )

    bytes_per_round = (
        bytes_per_client *
        clients_per_round
    )

    total_bytes = (
        bytes_per_round *
        config["num_rounds"]
    )

    return {

        "kb_per_client_per_round":
            round(bytes_per_client / 1024, 2),

        "kb_per_round":
            round(bytes_per_round / 1024, 2),

        "total_mb":
            round(total_bytes / (1024 * 1024), 2),
    }


# ==========================================================================
# FEDAVG
# ==========================================================================

def agg_fedavg(g_hats: np.ndarray):

    return g_hats.mean(axis=0)


# ==========================================================================
# STRATEGY
# ==========================================================================

class FedAvgStrategy(fl.server.strategy.Strategy):

    def __init__(
        self,
        config,
        data_manager,
        byzantine_assignments,
        input_dim,
    ):

        super().__init__()

        self.config = config

        self.data_manager = data_manager

        self.byzantine_assignments = (
            byzantine_assignments
        )

        self.input_dim = input_dim

        self.model = build_model(
            input_dim=input_dim,
            num_classes=config["num_classes"],
        )

        self.global_weights = (
            self._flatten_model()
        )

        self.d = len(self.global_weights)

        X_test, y_test = (
            data_manager.load_global_test_data()
        )

        self.test_dataset = NBaiotDataset(
            X_test,
            y_test,
        )

        self.round_metrics = []

    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager):

        return self._to_params(
            self.global_weights
        )

    def configure_fit(
        self,
        server_round,
        parameters,
        client_manager,
    ):

        n_sample = max(
            10,
            int(
                self.config["n_clients"] *
                self.config["fraction_fit"]
            )
        )

        clients = client_manager.sample(
            num_clients=n_sample,
            min_num_clients=10,
        )

        fit_ins = fl.common.FitIns(
            parameters,
            {"server_round": server_round},
        )

        return [(c, fit_ins) for c in clients]

    def configure_evaluate(
        self,
        server_round,
        parameters,
        client_manager,
    ):
        return []

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):
        return None, {}

    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):

        if not results:
            return None, {}

        g_list = []

        for proxy, fit_res in results:

            arrays = parameters_to_ndarrays(
                fit_res.parameters
            )

            g_hat = np.concatenate([
                a.ravel()
                for a in arrays
            ])

            g_list.append(g_hat)

        g_hats = np.vstack(g_list)

        g_t = agg_fedavg(g_hats)

        self.global_weights = (
            self.global_weights
            + self.config["global_lr"] * g_t
        )

        self._load_weights(
            self.global_weights
        )

        return self._to_params(
            self.global_weights
        ), {}

    # ------------------------------------------------------------------

    def evaluate(self, server_round, parameters):

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(device)

        self.model.eval()

        loader = DataLoader(
            self.test_dataset,
            batch_size=512,
            shuffle=False,
        )

        correct = 0
        total = 0

        with torch.no_grad():

            for x, y in loader:

                x = x.to(device)
                y = y.to(device)

                pred = torch.argmax(
                    self.model(x),
                    dim=1,
                )

                correct += (
                    pred == y
                ).sum().item()

                total += y.size(0)

        acc = correct / total if total > 0 else 0.0

        print(
            f"[FedAvg] "
            f"Round {server_round:3d} | "
            f"Acc={acc:.4f}"
        )

        self.round_metrics.append({

            "round": server_round,

            "accuracy":
                round(float(acc), 4),
        })

        return float(1 - acc), {
            "accuracy": float(acc)
        }

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _flatten_model(self):

        return np.concatenate([
            p.detach().cpu().numpy().ravel()
            for p in self.model.parameters()
            if p.requires_grad
        ])

    def _load_weights(self, flat):

        offset = 0

        with torch.no_grad():

            for p in self.model.parameters():

                if p.requires_grad:

                    size = p.numel()

                    p.copy_(torch.tensor(
                        flat[
                            offset:
                            offset + size
                        ].reshape(p.shape),

                        dtype=torch.float32,
                    ))

                    offset += size

    def _to_params(self, flat):

        arrays = []

        offset = 0

        for p in self.model.parameters():

            if p.requires_grad:

                size = p.numel()

                arrays.append(
                    flat[
                        offset:
                        offset + size
                    ].reshape(p.shape)
                )

                offset += size

        return ndarrays_to_parameters(
            arrays
        )


# ==========================================================================
# CLIENT FACTORY
# ==========================================================================

def make_client_fn(
    data_manager,
    byzantine_assignments,
    config,
    input_dim,
):

    def client_fn(context):

        cid = (
            int(context.node_config["partition-id"])
            % config["n_clients"]
        )

        X, y = data_manager.load_client_data(cid)

        dataset = NBaiotDataset(X, y)

        return NBaiotClient(

            client_id      = cid,

            dataset        = dataset,

            input_dim      = input_dim,

            num_classes    = config["num_classes"],

            local_epochs   = config["local_epochs"],

            local_lr       = config["local_lr"],

            batch_size     = config["batch_size"],

            is_byzantine   = (
                cid in byzantine_assignments
            ),

            attack_type    = (
                byzantine_assignments.get(cid, None)
            ),

            top_k_ratio    = 1.0,

            residual_decay = 0.9,

        ).to_client()

    return client_fn


# ==========================================================================
# FINAL EVALUATION
# ==========================================================================

def evaluate_final(
    flat_weights,
    data_manager,
    input_dim,
    config,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = build_model(
        input_dim=input_dim,
        num_classes=config["num_classes"],
    )

    offset = 0

    with torch.no_grad():

        for p in model.parameters():

            if p.requires_grad:

                size = p.numel()

                p.copy_(torch.tensor(
                    flat_weights[
                        offset:
                        offset + size
                    ].reshape(p.shape),

                    dtype=torch.float32,
                ))

                offset += size

    model.to(device)

    model.eval()

    X_test, y_test = (
        data_manager.load_global_test_data()
    )

    loader = DataLoader(
        NBaiotDataset(X_test, y_test),
        batch_size=512,
        shuffle=False,
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            preds = torch.argmax(
                model(x),
                dim=1,
            ).cpu().numpy()

            all_preds.extend(
                preds.tolist()
            )

            all_labels.extend(
                y.numpy().tolist()
            )

    all_preds = np.array(all_preds)

    all_labels = np.array(all_labels)

    n_classes = config["num_classes"]

    # ------------------------------------------------------------------
    # PER-CLASS F1
    # ------------------------------------------------------------------

    f1_per_class = f1_score(
        all_labels,
        all_preds,

        average=None,

        zero_division=0,

        labels=list(range(n_classes)),
    ).tolist()

    # ------------------------------------------------------------------
    # PER-CLASS RECALL
    # ------------------------------------------------------------------

    recall_per_class = recall_score(
        all_labels,
        all_preds,

        average=None,

        zero_division=0,

        labels=list(range(n_classes)),
    ).tolist()

    # ------------------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------------------

    conf_matrix = confusion_matrix(
        all_labels,
        all_preds,
    ).tolist()

    # ------------------------------------------------------------------

    report = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    return {

        "accuracy":
            round(float(
                accuracy_score(
                    all_labels,
                    all_preds,
                )
            ), 4),

        "f1_macro":
            round(float(
                f1_score(
                    all_labels,
                    all_preds,
                    average="macro",
                    zero_division=0,
                )
            ), 4),

        "precision_macro":
            round(float(
                precision_score(
                    all_labels,
                    all_preds,
                    average="macro",
                    zero_division=0,
                )
            ), 4),

        "recall_macro":
            round(float(
                recall_score(
                    all_labels,
                    all_preds,
                    average="macro",
                    zero_division=0,
                )
            ), 4),

        "f1_per_class":
            [round(v, 4) for v in f1_per_class],

        "recall_per_class":
            [round(v, 4) for v in recall_per_class],

        "confusion_matrix":
            conf_matrix,

        "classification_report":
            report,
    }


# ==========================================================================
# RUN ONE SCENARIO
# ==========================================================================

def run_scenario(
    alpha,
    attack_type,
    byz_fraction,
    config,
):

    print("\n" + "=" * 75)

    print(
        f"Alpha={alpha} | "
        f"Attack={attack_type} | "
        f"Byzantine={byz_fraction}"
    )

    print("=" * 75)

    data_dir = (
        f"data/partitioned/alpha_{alpha}_N200"
    )

    dm = DataManager(data_dir=data_dir)

    byzantine_assignments = (
        assign_single_attack_roles(
            config["n_clients"],
            byz_fraction,
            attack_type,
            config["seed"],
        )
    )

    strategy = FedAvgStrategy(
        config=config,
        data_manager=dm,
        byzantine_assignments=(
            byzantine_assignments
        ),
        input_dim=INPUT_DIM,
    )

    client_fn = make_client_fn(
        dm,
        byzantine_assignments,
        config,
        INPUT_DIM,
    )

    start = time.time()

    fl.simulation.start_simulation(

        client_fn=client_fn,

        num_clients=config["n_clients"],

        config=fl.server.ServerConfig(
            num_rounds=config["num_rounds"]
        ),

        strategy=strategy,

        client_resources={
            "num_cpus": 2,
            "num_gpus": 0.0,
        },
    )

    total_time = time.time() - start

    final_metrics = evaluate_final(
        strategy.global_weights,
        dm,
        INPUT_DIM,
        config,
    )

    # ------------------------------------------------------------------
    # COMMUNICATION
    # ------------------------------------------------------------------

    comm = compute_comm_cost(
        strategy.d,
        config,
    )

    # ------------------------------------------------------------------
    # CONVERGENCE ANALYSIS
    # ------------------------------------------------------------------

    per_round_acc = [
        m["accuracy"]
        for m in strategy.round_metrics
    ]

    best_acc = (
        max(per_round_acc)
        if per_round_acc
        else final_metrics["accuracy"]
    )

    best_round = (
        per_round_acc.index(best_acc) + 1
        if per_round_acc
        else config["num_rounds"]
    )

    attack_success_drop = round(
        1.0 - final_metrics["accuracy"],
        4,
    )

    # ------------------------------------------------------------------

    result = {

        "alpha":
            alpha,

        "attack_type":
            attack_type,

        "byzantine_fraction":
            byz_fraction,

        "accuracy":
            final_metrics["accuracy"],

        "best_accuracy":
            round(best_acc, 4),

        "best_round":
            best_round,

        "attack_success_drop":
            attack_success_drop,

        "f1_macro":
            final_metrics["f1_macro"],

        "precision_macro":
            final_metrics["precision_macro"],

        "recall_macro":
            final_metrics["recall_macro"],

        "f1_per_class":
            final_metrics["f1_per_class"],

        "recall_per_class":
            final_metrics["recall_per_class"],

        "confusion_matrix":
            final_metrics["confusion_matrix"],

        "classification_report":
            final_metrics["classification_report"],

        "comm_kb_per_round":
            comm["kb_per_round"],

        "comm_total_mb":
            comm["total_mb"],

        "total_time_s":
            round(total_time, 2),

        "time_per_round_s":
            round(
                total_time /
                config["num_rounds"],
                2,
            ),

        "per_round_accuracy":
            per_round_acc,
    }

    print("\nDONE")

    print(
        f"Accuracy      : "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Best Accuracy : "
        f"{result['best_accuracy']:.4f} "
        f"(round {best_round})"
    )

    print(
        f"F1 Macro      : "
        f"{result['f1_macro']:.4f}"
    )

    print(
        f"Attack Drop   : "
        f"{result['attack_success_drop']:.4f}"
    )

    return result


# ==========================================================================
# MAIN
# ==========================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_CONFIG["num_rounds"],
    )

    return parser.parse_args()


def main():

    args = parse_args()

    config = {**DEFAULT_CONFIG}

    config["num_rounds"] = args.rounds

    np.random.seed(config["seed"])

    torch.manual_seed(config["seed"])

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    all_results = []

    print("\n" + "=" * 75)
    print("FedAvg — FINAL Single Attack Benchmark")
    print("=" * 75)

    for alpha in ALPHAS:

        for attack_type in ATTACK_TYPES:

            for byz_fraction in BYZ_FRACTIONS:

                if ray.is_initialized():
                    ray.shutdown()

                ray.init(
                    num_cpus=4,
                    object_store_memory=(
                        512 * 1024 * 1024
                    ),
                )

                try:

                    result = run_scenario(
                        alpha,
                        attack_type,
                        byz_fraction,
                        config,
                    )

                    all_results.append(
                        result
                    )

                except Exception as e:

                    import traceback

                    print(
                        f"\n[ERROR] "
                        f"alpha={alpha} "
                        f"attack={attack_type} "
                        f"byz={byz_fraction}"
                    )

                    traceback.print_exc()

    # ------------------------------------------------------------------
    # SAVE JSON
    # ------------------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    save_path = os.path.join(
        RESULTS_DIR,
        f"fedavg_single_attacks_final_{timestamp}.json"
    )

    with open(save_path, "w") as f:

        json.dump(
            all_results,
            f,
            indent=2,
            default=str,
        )

    # ------------------------------------------------------------------

    print("\n" + "=" * 75)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 75)

    print(f"\nResults saved to:\n{save_path}")


if __name__ == "__main__":
    main()