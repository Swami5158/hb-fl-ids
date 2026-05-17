import numpy as np
from typing import List


# ---------------------------------------------------------------------------
# Algorithm 1 — Federated Chi-square Feature Selection
# ---------------------------------------------------------------------------

C_BINS: int = 10   # fixed globally so all clients discretize identically

# Fixed bin range for StandardScaler-normalized features.
# After StandardScaler, features are approximately N(0,1).
# [-4, 4] covers 99.99% of the distribution for all clients uniformly.
# This ensures bin 3 on client A and bin 3 on client B represent the
# same value range — required for meaningful score aggregation across
# non-IID clients with different local data distributions.
SCALED_BIN_MIN: float = -4.0
SCALED_BIN_MAX: float =  4.0


# ---------------------------------------------------------------------------
# Client side — compute local Chi-square scores
# ---------------------------------------------------------------------------

def compute_local_chisquare(
    X_raw: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    c_bins: int = C_BINS,
) -> np.ndarray:
    """
    Algorithm 1 Lines 3-9:
        For each feature v:
            chi2_i(v) = sum_c sum_j (O_icj - E_icj)^2 / E_icj

    NOTE: X_raw must be StandardScaler-normalized before calling this
    function. Fixed bins [-4, 4] are used so that bin boundaries are
    identical across all clients regardless of local data distribution.
    This is critical for non-IID settings where local value ranges differ.

    Vectorized implementation — no Python loops over samples.
    np.add.at builds the observed frequency table in one operation,
    making this ~50x faster than the naive loop version.

    Parameters
    ----------
    X_raw       : (n_samples, F) StandardScaler-normalized features
    y           : (n_samples,) integer class labels in range [0, num_classes-1]
    num_classes : J = 9 (after class merging)
    c_bins      : C = 10 (fixed globally)

    Returns
    -------
    chi2_scores : (F,) — only this is sent to server, no raw data leaves
    """
    n_samples, n_features = X_raw.shape

    if n_samples < 5:
        return np.zeros(n_features, dtype=np.float64)

    chi2_scores = np.zeros(n_features, dtype=np.float64)

    # Clip y to valid class range — safety against label encoding mismatches
    y_clipped = np.clip(y, 0, num_classes - 1)

    # Fixed bin edges — identical for ALL clients
    # Ensures comparable discretization across non-IID data distributions
    bin_edges = np.linspace(SCALED_BIN_MIN, SCALED_BIN_MAX, c_bins + 1)

    for v in range(n_features):
        feature_col = X_raw[:, v]

        # Skip constant or near-constant features — no discriminative power
        if np.var(feature_col) < 1e-10:
            continue

        # Discretize into C equal-width bins using fixed global boundaries
        bin_indices = np.digitize(feature_col, bin_edges[1:-1])  # 0-based
        bin_indices = np.clip(bin_indices, 0, c_bins - 1)        # safety clip

        # ------------------------------------------------------------------
        # Build observed frequency table O: (C, J)
        # np.add.at is ~50x faster than the Python for-loop version
        # ------------------------------------------------------------------
        O = np.zeros((c_bins, num_classes), dtype=np.float64)
        np.add.at(O, (bin_indices, y_clipped), 1)

        # Expected frequency table E under independence assumption
        row_totals = O.sum(axis=1, keepdims=True)   # (C, 1)
        col_totals = O.sum(axis=0, keepdims=True)   # (1, J)
        E = (row_totals * col_totals) / (n_samples + 1e-10)

        # Chi-square statistic — skip cells where E = 0
        mask = E > 0
        chi2_scores[v] = np.sum(((O[mask] - E[mask]) ** 2) / E[mask])

    return chi2_scores   # (F,) — only this crosses the client boundary


# ---------------------------------------------------------------------------
# Server side — aggregate and select
# ---------------------------------------------------------------------------

def aggregate_chisquare_scores(
    local_scores: List[np.ndarray],
) -> np.ndarray:
    """
    Algorithm 1 Line 14:
        chi2_global(v) = (1/N) * sum_i chi2_i(v)
    """
    if len(local_scores) == 0:
        raise ValueError("No client scores received.")
    stacked = np.vstack(local_scores)      # (N, F)
    return stacked.mean(axis=0)            # (F,)


def select_top_features(
    chi2_global: np.ndarray,
    fsel: int = 70,
) -> np.ndarray:
    """
    Algorithm 1 Lines 16-19:
        Rank features descending by chi2_global(v)
        S = top Fsel indices

    Safety clamp applied BEFORE computing S to avoid index errors
    when fsel > number of available features.
    """
    # Clamp fsel FIRST — must happen before ranked[:fsel] slicing
    fsel = max(1, min(fsel, len(chi2_global)))

    ranked    = np.argsort(chi2_global)[::-1]
    S         = np.sort(ranked[:fsel])        # ascending order for clean indexing
    threshold = chi2_global[ranked[fsel - 1]]

    print(f"[FeatureSelection] Selected {fsel} / {len(chi2_global)} features")
    print(f"[FeatureSelection] Score range: "
          f"{chi2_global[S].min():.4f} → {chi2_global[S].max():.4f}")
    print(f"[FeatureSelection] Threshold (rank {fsel}): {threshold:.4f}")

    return S


# ---------------------------------------------------------------------------
# Full Algorithm 1 orchestrator
# ---------------------------------------------------------------------------

def run_federated_feature_selection(
    data_manager,
    client_ids: List[int],
    fsel: int = 70,
    num_classes: int = 9,
    c_bins: int = C_BINS,
) -> np.ndarray:
    """
    Runs full Algorithm 1 end-to-end:
        Phase 1 (client side) : each client computes local Chi-square scores
                                 on StandardScaler-normalized features with
                                 fixed global bin boundaries
        Phase 2 (server side) : aggregate → rank → select top Fsel features
        Broadcast S           : apply to DataManager

    Key design decisions:
        - Scaler applied BEFORE chi-square so bins are comparable across clients
        - Fixed bins [-4, 4] ensure identical discretization for all clients
        - Only chi2 scores (shape F,) leave each client — no raw data shared
        - num_classes=9 after merging indistinguishable class pairs

    Parameters
    ----------
    data_manager : DataManager instance (scaler already fitted)
    client_ids   : list of integer client IDs (0 to N-1)
    fsel         : Fsel = 70 features to select
    num_classes  : J = 9 (after class merge)
    c_bins       : C = 10 bins per feature

    Returns
    -------
    S : (Fsel,) selected feature index array (sorted ascending)
    """
    print("\n" + "=" * 60)
    print("Algorithm 1: Federated Chi-square Feature Selection")
    print(f"  num_classes={num_classes}, fsel={fsel}, c_bins={c_bins}")
    print(f"  Bin range: [{SCALED_BIN_MIN}, {SCALED_BIN_MAX}] (fixed global bins)")
    print("=" * 60)

    local_scores: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Phase 1: Client side — only scores leave each client
    # ------------------------------------------------------------------
    for i, client_id in enumerate(client_ids):
        print(f"  [Client {client_id}] Computing Chi-square "
              f"({i+1}/{len(client_ids)})...")

        # Load raw unscaled features + merged labels
        X_raw, y = data_manager.load_client_data_raw(client_id)

        if len(X_raw) == 0:
            print(f"  [Client {client_id}] Skipped — no data.")
            continue

        # Scale using the globally fitted scaler BEFORE computing chi-square
        # This ensures fixed bins [-4, 4] are meaningful for all clients
        X_scaled = data_manager.scaler.transform(X_raw)

        scores = compute_local_chisquare(
            X_raw=X_scaled,      # scaled features — fixed bins apply
            y=y,
            num_classes=num_classes,
            c_bins=c_bins,
        )

        local_scores.append(scores)
        top3 = np.argsort(scores)[::-1][:3].tolist()
        print(f"  [Client {client_id}] Done. Top-3 features: {top3}")

    # ------------------------------------------------------------------
    # Phase 2: Server side — never sees raw data
    # ------------------------------------------------------------------
    print("\n  [Server] Aggregating scores...")
    chi2_global = aggregate_chisquare_scores(local_scores)

    print("  [Server] Selecting top features...")
    S = select_top_features(chi2_global, fsel=fsel)

    # Broadcast S to DataManager — applied to all subsequent data loading
    data_manager.apply_feature_selection(S)
    print(f"\n  [Server] S broadcast complete — "
          f"first 10 indices: {S[:10].tolist()}")
    print("=" * 60 + "\n")

    return S
