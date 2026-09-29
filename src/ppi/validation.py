"""Holdout evaluation: expert PPI vs linear surrogate."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .models import PositionModelBundle
from .scoring import compute_ppi, display_scale_from_raw, scale_features


def _surrogate_display_ppi(
    match_df: pd.DataFrame,
    bundle: PositionModelBundle,
) -> np.ndarray:
    """Map surrogate raw predictions to 0–10 using the same cohort rescale as expert PPI."""
    raw_preds = np.array(
        [bundle.predict_raw(str(row["profile"]), row) for _, row in match_df.iterrows()],
        dtype=float,
    )
    return display_scale_from_raw(raw_preds, display_scale=bundle.display_scale)


def compare_expert_vs_surrogate(
    match_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    bundle: PositionModelBundle,
) -> pd.DataFrame:
    """
    Compare expert and surrogate PPI on real match rows (held out from training).

    Expert PPI uses within-match cohort rescale. Surrogate raw scores are rescaled
    with the same display rule for a fair side-by-side.
    """
    expert_scored, _, _ = compute_ppi(match_df, weights_df, display_scale=bundle.display_scale)
    surr_display = _surrogate_display_ppi(match_df, bundle)

    out = match_df[["name", "profile"]].copy()
    out["expert_ppi"] = expert_scored["ppi"].values
    out["surrogate_ppi"] = np.round(surr_display, 2)
    out["delta"] = np.round(out["surrogate_ppi"] - out["expert_ppi"], 2)
    return out.sort_values("expert_ppi", ascending=False).reset_index(drop=True)


def holdout_metrics(
    match_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    bundle: PositionModelBundle,
) -> dict[str, float]:
    """Aggregate error metrics on a holdout match squad."""
    expert_scored, _, _ = compute_ppi(match_df, weights_df, display_scale=bundle.display_scale)
    y_true = expert_scored["ppi"].values
    y_pred = _surrogate_display_ppi(match_df, bundle)

    rho = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return {
        "n": float(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(rho) if rho is not None and not np.isnan(rho) else float("nan"),
    }
