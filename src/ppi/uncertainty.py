"""Bootstrap / conformal-style uncertainty on match PPI rankings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import FEATURE_COLUMNS
from .scoring import compute_ppi, display_scale_from_raw
from .weights import weight_vector


def bootstrap_ppi_intervals(
    match_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    *,
    n_boot: int = 400,
    alpha: float = 0.1,
    random_state: int = 42,
    display_scale: float = 10.0,
    min_minutes: float = 0.0,
) -> pd.DataFrame:
    """
    Percentile bootstrap of display PPI by resampling the match cohort.

    Each draw: sample players with replacement → refit MinMax on the draw →
    score original players with those scale limits → cohort-rescale to 0–10.
    This captures ranking uncertainty from scale/cohort composition.
    """
    frame = match_df.copy()
    if min_minutes > 0 and "playing_time" in frame.columns:
        frame = frame.loc[frame["playing_time"] >= min_minutes].reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()

    n = len(frame)
    rng = np.random.default_rng(random_state)
    point, scaler, _ = compute_ppi(frame, weights_df, display_scale=display_scale)
    point_scores = point["ppi"].values

    boot = np.zeros((n_boot, n), dtype=float)
    profiles = frame["profile"].tolist()
    X = frame[list(FEATURE_COLUMNS)].astype(float).values

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        X_boot = X[idx]
        # Fit scale on bootstrap sample
        mins = X_boot.min(axis=0)
        maxs = X_boot.max(axis=0)
        span = np.where(np.isclose(maxs, mins), 1.0, maxs - mins)
        X_scaled = (X - mins) / span
        X_scaled = np.clip(X_scaled, 0.0, 1.0)

        raw = np.zeros(n, dtype=float)
        for i, profile in enumerate(profiles):
            betas = np.asarray(weight_vector(weights_df, profile), dtype=float)
            raw[i] = float(np.dot(betas, X_scaled[i]))
        boot[b] = display_scale_from_raw(raw, display_scale=display_scale)

    lo = np.quantile(boot, alpha / 2, axis=0)
    hi = np.quantile(boot, 1 - alpha / 2, axis=0)
    # Rank stability: share of draws where player is top-3
    top3_rate = (np.argsort(-boot, axis=1).argsort(axis=1) < 3).mean(axis=0)

    out = frame[["name", "profile"]].copy()
    out["ppi"] = np.round(point_scores, 2)
    out["ppi_lo"] = np.round(lo, 2)
    out["ppi_hi"] = np.round(hi, 2)
    out["top3_prob"] = np.round(top3_rate, 3)
    out["width"] = np.round(out["ppi_hi"] - out["ppi_lo"], 2)
    out = out.sort_values("ppi", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def conformal_display_band(
    residuals: np.ndarray,
    point: float,
    *,
    alpha: float = 0.1,
) -> tuple[float, float]:
    """Split-conformal absolute residual band around a point estimate."""
    if len(residuals) == 0:
        return point, point
    q = float(np.quantile(np.abs(residuals), 1 - alpha, method="higher"))
    return max(0.0, point - q), min(10.0, point + q)
