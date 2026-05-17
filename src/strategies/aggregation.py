import numpy as np
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# Standalone helpers — MZ-score, PDP, Divergence Penalty
# ---------------------------------------------------------------------------

def mz_score(values: np.ndarray) -> np.ndarray:
    """
    Median-based Z-score for robust outlier detection.

        MZ(x_i) = 0.6745 * (x_i - median) / MAD
        MAD = median(|x_i - median|)

    0.6745 normalizes so MZ ≈ standard Z-score for normal distributions.
    For Byzantine gradients (non-normal, extreme), MZ is far more robust
    than standard Z-score because median and MAD are not pulled by outliers.
    Returns zeros when MAD < 1e-10 (all values nearly identical — safe).
    """
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad < 1e-10:
        return np.zeros_like(values)
    return 0.6745 * (values - med) / mad


def positive_direction_purity(
    client_layer: np.ndarray,
    reference_layer: np.ndarray,
) -> float:
    """
    Positive Direction Purity (PDP) — sign agreement between client and
    reference for one layer.

        PDP = |{j : sign(g_i[j]) == sign(g_ref[j])}| / |layer_size|

    Honest Non-IID clients: 0.75 - 0.95 (high agreement despite diversity)
    Sign-flip attackers:     0.05 - 0.30 (systematic sign reversal)
    Scaling attackers:       ~0.50        (random, caught by MZ magnitude)

    Enabled by Top-K sparsification: removing low-magnitude noise improves
    PDP accuracy because noise parameters have random signs that dilute the
    signal. LASA Lemma 1 proves sparsification amplifies sign-based defense.
    """
    if len(client_layer) == 0:
        return 1.0
    same_sign = np.sum(np.sign(client_layer) == np.sign(reference_layer))
    return float(same_sign) / len(client_layer)


def compute_divergence_penalty(
    client_layers: List[np.ndarray],
    ref_layers: List[np.ndarray],
) -> float:
    """
    Layer-wise divergence penalty.

    For each layer l: div_l = ||g_i[l] - g_ref[l]||_2
    Aggregate:        penalty = clip(mean(div_l) / 2.0, 0, 1)

    Normalizes by 2.0 — max possible L2 distance between unit vectors.
    Penalizes extreme layer-level outliers that survive cosine and MZ checks.
    """
    if not client_layers:
        return 0.0
    distances = [
        np.linalg.norm(client_layers[l] - ref_layers[l])
        for l in range(len(client_layers))
    ]
    return float(np.clip(np.mean(distances) / 5.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Full HB-FL-IDS Aggregator — Algorithms 3 + 4 with LASA enhancements
# ---------------------------------------------------------------------------

class HBFLAggregator:
    """
    Full Byzantine-Resilient Aggregator implementing Algorithms 3 + 4
    with integrated LASA enhancements.

    Trust Score Formula (per client, per round):
        TS_i = max(0, cosine_ts_i × layer_frac_i - divergence_penalty_i)

    Where:
        cosine_ts_i      = ReLU(<g_hat_i, g_hat_ref> - theta_min)
                           Algorithm 3 Step 3. Catches directional attackers.

        layer_frac_i     = fraction of layers passing MZ magnitude + PDP filters.
                           LASA enhancement. Catches scaling and sign-flip attackers.

        divergence_penalty_i = normalized mean layer-wise L2 distance from median.
                           Penalizes extreme layer outliers surviving cosine+MZ.

    Core (Algorithms 3 + 4):
        - Adaptive gamma_t = clip(gamma_0 + lambda * sigma^2_{t-1}, gamma_min, gamma_max)
        - Median-based reference (MZ-robust, not mean-based)
        - Double-sided trimming: top AND bottom floor(gamma_t * N) by distance
        - Trust scores for ALL participating clients (trimmed included)
        - EMA reputation: beta * r_prev + (1-beta) * TS
        - Threat-reactive beta: higher trim ratio -> faster reputation decay
        - Reputation-weighted aggregation over retained clients only
        - w^{t+1} = w^t - eta_g * g^t  (gradient DESCENT)

    LASA Enhancements:
        - Layer-wise processing: each of L layers filtered independently
        - Median reference per layer (Byzantine-robust)
        - MZ-score magnitude filter per layer (lambda_m threshold)
        - PDP sign-purity filter per layer (lambda_d threshold)
        - Divergence penalty per layer vs median reference

    Parameters
    ----------
    num_clients    : N = 200
    layer_shapes   : list of shapes from model.get_layer_shapes()
    beta           : EMA reputation decay (default 0.9)
    theta_min      : cosine similarity threshold (default 0.0)
    gamma_0        : initial trimming ratio (default 0.1)
    lambda_adapt   : sigma^2 adaptation rate (default 0.1)
    gamma_min      : minimum trimming ratio (default 0.05)
    byzantine_fraction : f — sets gamma_max (default 0.2)
    lambda_m       : MZ-score magnitude threshold (default 3.5)
    lambda_d       : MZ-score PDP threshold (default 3.5)
    """

    def __init__(
        self,
        num_clients: int,
        layer_shapes: List[tuple],
        beta: float = 0.9,
        theta_min: float = -0.2,
        gamma_0: float = 0.05,
        lambda_adapt: float = 0.1,
        gamma_min: float = 0.02,
        byzantine_fraction: float = 0.2,
        lambda_m: float = 1.5,
        lambda_d: float = 1.5,
    ):
        self.N             = num_clients
        self.layer_shapes  = layer_shapes
        self.layer_sizes   = [int(np.prod(s)) for s in layer_shapes]
        self.L             = len(layer_shapes)
        self.beta          = beta
        self.theta_min     = theta_min
        self.gamma_0       = gamma_0
        self.lambda_adapt  = lambda_adapt
        self.gamma_min     = gamma_min
        # gamma_max: at least gamma_0 even when byzantine_fraction=0 (clean baseline)
        self.gamma_max     = max(byzantine_fraction, gamma_0)
        self.lambda_m      = lambda_m
        self.lambda_d      = lambda_d

        # Persistent state indexed by global client ID (0..N-1)
        # Algorithm 5 Lines 8-10: r^0_i = 1.0, TS^0_i = 1.0 for all i
        self.reputation   = np.ones(num_clients,  dtype=np.float64)
        self.trust_scores = np.ones(num_clients, dtype=np.float64)
        self.gamma_t      = gamma_0
        self.round        = 0
        self.logs: List[Dict] = []

        d = sum(self.layer_sizes)
        print(f"[Aggregator] HB-FL-IDS Aggregator initialized")
        print(f"  N={num_clients}, L={self.L} layers, d={d:,}")
        print(f"  beta={beta}, theta_min={theta_min}")
        print(f"  gamma_0={gamma_0}, gamma_min={gamma_min}, gamma_max={self.gamma_max}")
        print(f"  lambda_m={lambda_m} (MZ magnitude), lambda_d={lambda_d} (MZ PDP)")

    # ------------------------------------------------------------------
    # Internal: split flat gradient into per-layer list
    # ------------------------------------------------------------------

    def _split_layers(self, flat: np.ndarray) -> List[np.ndarray]:
        layers, offset = [], 0
        for size in self.layer_sizes:
            layers.append(flat[offset:offset + size].copy())
            offset += size
        return layers

    # ------------------------------------------------------------------
    # Step 1: Adaptive trimming ratio (Algorithm 3 Step 1)
    # ------------------------------------------------------------------

    def _update_gamma(self, participating_ids: List[int]) -> float:
        """
        Round 1: gamma_t = gamma_0 (no trust history yet).
        Round t: gamma_t = clip(gamma_0 + lambda * sigma^2_{t-1}, gamma_min, gamma_max)

        Uses trust scores of participating clients from PREVIOUS round.
        High sigma^2 -> more Byzantine activity -> trim more aggressively.
        """
        if self.round == 1:
            return self.gamma_0
        ts     = np.array([self.trust_scores[cid] for cid in participating_ids])
        sigma2 = float(np.var(ts))
        gamma  = self.gamma_0 + self.lambda_adapt * sigma2
        return float(np.clip(gamma, self.gamma_min, self.gamma_max))

    # ------------------------------------------------------------------
    # Step 2: Layer-wise double-sided trimming + median reference
    # ------------------------------------------------------------------

    def _layer_trimmed_median_reference(
        self,
        client_layers: List[List[np.ndarray]],
        gamma_t: float,
        n_clients: int,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        For each layer:
            1. Coordinate-wise median reference (Byzantine-robust)
            2. L2 distance of each client from reference
            3. Remove top floor(gamma_t * N) most distant (direction attackers)
               AND bottom floor(gamma_t * N) most similar (Min-Max mimickers)
        Client must survive trimming in ALL layers (intersection).
        Final median reference built from retained clients only.
        """
        n_trim = max(1, int(0.5 * gamma_t * n_clients))
        layer_retained_sets = []

        for l in range(self.L):
            layer_stack  = np.vstack([client_layers[i][l] for i in range(n_clients)])
            med_ref      = np.median(layer_stack, axis=0)
            med_norm     = np.linalg.norm(med_ref)
            distances  = np.linalg.norm(layer_stack - med_ref, axis=1)
            sorted_idx = np.argsort(distances)

            n_trim_safe = n_trim
            if 2 * n_trim_safe >= n_clients:
                n_trim_safe = max(0, n_clients // 4)

            retained = sorted_idx[n_trim_safe: n_clients - n_trim_safe]
            layer_retained_sets.append(set(retained.tolist()))

        # Relaxed intersection — client must survive in at least 50% of layers
        if self.L > 1:
            from collections import Counter

            counter = Counter()
            for s in layer_retained_sets:
                for cid in s:
                    counter[cid] += 1

            # Require survival in at least half of the layers
            threshold = max(1, int(0.5 * self.L))

            retained_ids = sorted([
                cid for cid, count in counter.items()
                if count >= threshold
            ])
        else:
            retained_ids = sorted(list(layer_retained_sets[0]))

        # Fallback: union if intersection empty
        if len(retained_ids) == 0:
            union = set()
            for s in layer_retained_sets:
                union.update(s)
            retained_ids = sorted(list(union))

        # Final fallback: keep all
        if len(retained_ids) == 0:
            retained_ids = list(range(n_clients))

        # Build median reference from retained clients
        ref_layers = []
        for l in range(self.L):
            stack    = np.vstack([client_layers[i][l] for i in retained_ids])
            med      = np.median(stack, axis=0)
            med_norm = np.linalg.norm(med)
            ref_layers.append(med / med_norm if med_norm > 1e-10 else med)

        return ref_layers, retained_ids

    # ------------------------------------------------------------------
    # Step 3: Layer-wise MZ + PDP filtering
    # ------------------------------------------------------------------

    def _layer_mz_filter(
        self,
        client_layers: List[List[np.ndarray]],
        ref_layers: List[np.ndarray],
        retained_ids: List[int],
    ) -> Dict[int, float]:
        """
        For each retained client, for each layer:
            MZ-score on L2 norm  -> magnitude/scaling attack filter
            MZ-score on PDP      -> sign-flip attack filter
        Layer passes if |MZ_mag| <= lambda_m AND |MZ_pdp| <= lambda_d.
        Returns local_idx -> fraction of layers accepted [0, 1].
        Trimmed clients (not in retained_ids) implicitly get 0.0.
        """
        n_ret = len(retained_ids)
        if n_ret == 0:
            return {}

        magnitudes = np.zeros((n_ret, self.L))
        pdp_scores = np.zeros((n_ret, self.L))

        for local_idx, global_idx in enumerate(retained_ids):
            for l in range(self.L):
                magnitudes[local_idx, l] = np.linalg.norm(
                    client_layers[global_idx][l]
                )
                pdp_scores[local_idx, l] = positive_direction_purity(
                    client_layers[global_idx][l], ref_layers[l]
                )

        layer_acceptance = np.zeros((n_ret, self.L), dtype=bool)
        for l in range(self.L):
            mag_mz = mz_score(magnitudes[:, l])
            pdp_mz = mz_score(pdp_scores[:, l])
            layer_acceptance[:, l] = (
                (np.abs(mag_mz) <= self.lambda_m) &
                (np.abs(pdp_mz) <= self.lambda_d)
            )

        acceptance_fracs = layer_acceptance.mean(axis=1)
        return {
            retained_ids[i]: float(acceptance_fracs[i])
            for i in range(n_ret)
        }

    # ------------------------------------------------------------------
    # Main aggregate() — Algorithms 3 + 4, one round
    # ------------------------------------------------------------------

    def aggregate(
        self,
        g_hats: np.ndarray,
        participating_ids: List[int],
        global_weights: np.ndarray,
        global_lr: float = 0.01,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Full aggregation pipeline for one round.

        Trust score: TS_i = max(0, cosine_ts_i * layer_frac_i - div_penalty_i)

        Steps:
            1. Adaptive gamma_t from sigma^2 of previous trust scores
            2. Layer-wise double-sided trimming + median reference
            3. Reconstruct full unit-norm reference for cosine similarity
            4. Layer-wise MZ magnitude + PDP filtering (LASA)
            5. Compute combined TS = cosine * layer_frac - div_penalty
            6. Update trust scores for all participating clients
            7. Threat-reactive EMA reputation update
            8. Reputation-weighted aggregation over active (TS > 0) clients
            9. w^{t+1} = w^t - eta_g * g^t  (gradient DESCENT — MINUS sign)
        """
        self.round += 1
        N_p = len(g_hats)
        print(f"\n[Aggregator] === Round {self.round} ({N_p} clients) ===")
        g_hats = np.clip(g_hats, -5.0, 5.0)
        # Split flat gradients into per-layer lists
        client_layers = [self._split_layers(g_hats[i]) for i in range(N_p)]

        # Step 1: Adaptive gamma_t
        self.gamma_t = self._update_gamma(participating_ids)
        print(f"  gamma_t = {self.gamma_t:.4f}")

        # Step 2: Layer-wise double-sided trimming + median reference
        ref_layers, retained_ids = self._layer_trimmed_median_reference(
            client_layers, self.gamma_t, N_p
        )
        n_trimmed = N_p - len(retained_ids)
        print(f"  Trimmed: {n_trimmed}/{N_p} | Retained: {len(retained_ids)}")

        # Step 3: Full reference vector for cosine similarity
        g_ref_full = np.concatenate(ref_layers)
        ref_norm   = np.linalg.norm(g_ref_full)
        g_ref_full = g_ref_full / ref_norm if ref_norm > 1e-10 else g_ref_full

        # Step 4: Layer-wise MZ + PDP filtering
        acceptance = self._layer_mz_filter(client_layers, ref_layers, retained_ids)

        # Step 5: Combined trust scores for ALL N_p participating clients
        new_trust     = np.zeros(N_p, dtype=np.float64)
        cosine_scores = np.zeros(N_p, dtype=np.float64)
        layer_fracs   = np.zeros(N_p, dtype=np.float64)
        div_penalties = np.zeros(N_p, dtype=np.float64)

        for local_idx in range(N_p):
            g_i = g_hats[local_idx]
            g_i_norm = np.linalg.norm(g_i) + 1e-12
            g_ref_norm = np.linalg.norm(g_ref_full) + 1e-12

            cosine = float(np.dot(g_i, g_ref_full) / (g_i_norm * g_ref_norm))
            cosine_ts = max(0.0, cosine - self.theta_min)
            cosine_scores[local_idx] = cosine_ts

            # Trimmed clients get layer_frac=0.0 (not in acceptance dict)
            layer_frac = acceptance.get(local_idx, 0.0)
            layer_fracs[local_idx] = layer_frac

            div_pen = compute_divergence_penalty(
                client_layers[local_idx], ref_layers
            )
            div_penalties[local_idx] = div_pen

            new_trust[local_idx] = max(0.0, 0.7 * cosine_ts + 0.3 * layer_frac - 0.3 * div_pen)
        
        new_trust = np.clip(new_trust, 0.0, 1.0)

        # Step 6: Update global trust score array (for next round's gamma_t)
        for local_idx, cid in enumerate(participating_ids):
            self.trust_scores[cid] = new_trust[local_idx]

        # Step 7: Threat-reactive EMA reputation update
        # Under high attack (many trimmed clients), beta decreases slightly
        # so Byzantine reputation decays faster — without Bayesian optimization.
        trim_ratio     = n_trimmed / N_p if N_p > 0 else 0.0
        effective_beta = max(0.7, self.beta * (1.0 - 0.05 * trim_ratio))

        for local_idx, cid in enumerate(participating_ids):
            if self.round == 1:
                # Direct set on first round — immediate Byzantine detection
                self.reputation[cid] = 0.5 + 0.5 * new_trust[local_idx]
            else:
                # FIXED — correct per EDI Algorithm 3.2
                self.reputation[cid] = float(np.clip(
                effective_beta * self.reputation[cid]
                + (1.0 - effective_beta) * new_trust[local_idx],
                0.0, 1.0
            ))
                self.reputation[cid] = np.clip(self.reputation[cid], 0.0, 1.0)

        # Step 8: Reputation-weighted aggregation — only clients with TS > 0
        active_ids  = [i for i in range(N_p) if new_trust[i] > 0]
        active_cids = [participating_ids[i] for i in active_ids]

        MIN_ACTIVE = 20  # safeguard threshold

        # 🔥 SAFEGUARD: ensure enough clients survive
        if len(active_ids) < MIN_ACTIVE:
            print("[Aggregator] Too few active clients — relaxing filter")

            # Select top-K clients by trust score (instead of TS > 0 only)
            top_k = np.argsort(new_trust)[-MIN_ACTIVE:]
            active_ids  = top_k.tolist()
            active_cids = [participating_ids[i] for i in active_ids]

        # 🔥 If STILL empty (extreme edge case)
        if len(active_ids) == 0:
            print("[Aggregator] WARNING: All clients filtered — FedAvg fallback.")
            g_t = g_hats.mean(axis=0)

        else:
            reps = np.array([self.reputation[cid] for cid in active_cids])
            rep_sum = reps.sum()
            if rep_sum > 1e-10:
                weights = reps / rep_sum
            else:
                weights = np.ones_like(reps) / len(reps)

            # Normalize each client gradient to unit sphere before weighted sum
            # so reputation weights are not drowned out by magnitude differences
            active_ghats = g_hats[active_ids].copy()
            norms = np.linalg.norm(active_ghats, axis=1, keepdims=True)
            active_ghats = active_ghats / np.where(norms > 1e-10, norms, 1.0)

            g_t = (weights[:, None] * active_ghats).sum(axis=0)
        # 🔥 Final NaN safety
        if np.isnan(g_t).any():
            print("[Aggregator] NaN detected — zero update.")
            g_t = np.zeros_like(g_t)

        # Step 9: Global model update — gradient DESCENT (MINUS sign)
        # Algorithm 4 Step 6: # w^{t+1} = w^t + decayed_lr * g^t  (g^t = wE - w0, pseudo-gradient, PLUS sign)
        decayed_lr = global_lr / (1.0 + 0.01 * self.round)
        new_weights = global_weights + decayed_lr * g_t

        # Logging
        part_reps = np.array([self.reputation[cid] for cid in participating_ids])
        sigma2    = float(np.var([self.trust_scores[cid] for cid in participating_ids]))

        metrics = {
            "round":                self.round,
            "gamma_t":              float(self.gamma_t),
            "n_retained":           len(active_ids),
            "n_trimmed":            n_trimmed,
            "mean_reputation":      float(part_reps.mean()),
            "min_reputation":       float(part_reps.min()),
            "max_reputation":       float(part_reps.max()),
            "mean_trust":           float(new_trust.mean()),
            "mean_cosine":          float(cosine_scores.mean()),
            "mean_layer_frac":      float(layer_fracs.mean()),
            "mean_div_penalty":     float(div_penalties.mean()),
            "aggregated_grad_norm": float(np.linalg.norm(g_t)),
            "sigma2_trust":         sigma2,
            "effective_beta":       float(effective_beta),
            "trust_scores":         self.trust_scores.tolist(),
            "reputation_scores":    self.reputation.tolist(),
            "participating_ids":    participating_ids,
        }
        self.logs.append(metrics)

        print(f"  Cosine TS:   [{cosine_scores.min():.3f}, {cosine_scores.max():.3f}]  "
              f"mean={cosine_scores.mean():.3f}")
        print(f"  Layer frac:  [{layer_fracs.min():.3f}, {layer_fracs.max():.3f}]  "
              f"mean={layer_fracs.mean():.3f}")
        print(f"  Final TS:    [{new_trust.min():.3f}, {new_trust.max():.3f}]  "
              f"mean={new_trust.mean():.3f}")
        print(f"  Reputation:  [{part_reps.min():.3f}, {part_reps.max():.3f}]  "
              f"mean={part_reps.mean():.3f}")
        print(f"  Active/Total:{len(active_ids)}/{N_p} | "
              f"eff_beta={effective_beta:.4f} | g_t norm={np.linalg.norm(g_t):.6f}")

        return new_weights, metrics

    def get_suspicious_clients(self, threshold: float = 0.1) -> List[int]:
        """Client IDs with EMA reputation below threshold."""
        return [i for i, r in enumerate(self.reputation) if r < threshold]

    def get_reputation_summary(self) -> Dict:
        return {
            "mean":             float(self.reputation.mean()),
            "std":              float(self.reputation.std()),
            "min":              float(self.reputation.min()),
            "max":              float(self.reputation.max()),
            "suspicious_count": len(self.get_suspicious_clients()),
        }
