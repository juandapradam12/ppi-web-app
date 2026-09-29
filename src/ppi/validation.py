"""Holdout evaluation: expert PPI vs surrogate variants."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .models import PositionModelBundle, Variant
from .scoring import compute_ppi


def _calibrated_display(match_df: pd.DataFrame, bundle: PositionModelBundle) -> np.ndarray:
    ridge_display = bundle.batch_display_predictions(match_df, variant="ridge")
    if bundle.calibrator is None:
        return ridge_display
    return bundle.calibrator.predict(ridge_display)


def holdout_metrics(
    match_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    bundle: PositionModelBundle,
    *,
    variant: Variant = "calibrated",
) -> dict[str, float]:
    expert_scored, _, _ = compute_ppi(match_df, weights_df, display_scale=bundle.display_scale)
    y_true = expert_scored["ppi"].values

    if variant == "ols":
        y_pred = bundle.batch_display_predictions(match_df, variant="ols")
    elif variant == "ridge":
        y_pred = bundle.batch_display_predictions(match_df, variant="ridge")
    else:
        y_pred = _calibrated_display(match_df, bundle)

    rho = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return {
        "n": float(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(rho) if rho is not None and not np.isnan(rho) else float("nan"),
        "variant": variant,
    }


def leave_one_match_out_metrics(
    holdout_frames: list[pd.DataFrame],
    weights_df: pd.DataFrame,
    bundle: PositionModelBundle,
) -> dict:
    """
    For each match, fit isotonic on the other matches' (ridge_display, expert) pairs,
    then evaluate calibrated predictions on the left-out match.
    """
    from sklearn.isotonic import IsotonicRegression

    per_match: dict[str, dict[str, float]] = {}
    all_true: list[float] = []
    all_pred: list[float] = []
    residuals: list[float] = []

    prepared = []
    for frame in holdout_frames:
        expert, _, _ = compute_ppi(frame, weights_df, display_scale=bundle.display_scale)
        ridge = bundle.batch_display_predictions(frame, variant="ridge")
        mid = str(frame["match_id"].iloc[0])
        prepared.append((mid, frame, expert["ppi"].values, ridge))

    for i, (mid, frame, y_true, ridge) in enumerate(prepared):
        xs, ys = [], []
        for j, (_, _, y_j, ridge_j) in enumerate(prepared):
            if i == j:
                continue
            xs.extend(ridge_j.tolist())
            ys.extend(y_j.tolist())
        if len(xs) < 5:
            y_pred = ridge
        else:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(np.array(xs), np.array(ys))
            y_pred = iso.predict(ridge)

        rho = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
        per_match[mid] = {
            "n": float(len(y_true)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "spearman": float(rho) if rho is not None and not np.isnan(rho) else float("nan"),
        }
        all_true.extend(y_true.tolist())
        all_pred.extend(np.asarray(y_pred).tolist())
        residuals.extend((np.asarray(y_pred) - y_true).tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    rho = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return {
        "per_match": per_match,
        "pooled": {
            "n": float(len(y_true)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "spearman": float(rho) if rho is not None and not np.isnan(rho) else float("nan"),
        },
        "pooled_residuals": residuals,
    }


def holdout_metrics_all_variants(
    holdout_frames: list[pd.DataFrame],
    weights_df: pd.DataFrame,
    bundle: PositionModelBundle,
) -> dict[str, dict[str, float]]:
    """Pooled metrics per variant across all holdout matches."""
    out: dict[str, dict[str, float]] = {}
    for variant in ("ols", "ridge", "calibrated"):
        ys_true: list[float] = []
        ys_pred: list[float] = []
        for frame in holdout_frames:
            expert, _, _ = compute_ppi(frame, weights_df, display_scale=bundle.display_scale)
            ys_true.extend(expert["ppi"].tolist())
            if variant == "ols":
                ys_pred.extend(bundle.batch_display_predictions(frame, variant="ols").tolist())
            elif variant == "ridge":
                ys_pred.extend(bundle.batch_display_predictions(frame, variant="ridge").tolist())
            else:
                ys_pred.extend(_calibrated_display(frame, bundle).tolist())
        y_true = np.array(ys_true)
        y_pred = np.array(ys_pred)
        rho = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
        out[variant] = {
            "n": float(len(y_true)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "spearman": float(rho) if rho is not None and not np.isnan(rho) else float("nan"),
        }
    return out


def compare_expert_vs_surrogate(
    match_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    bundle: PositionModelBundle,
) -> pd.DataFrame:
    """Backward-compatible: expert vs calibrated surrogate."""
    full = compare_expert_vs_variants(match_df, weights_df, bundle)
    return full[
        ["name", "profile", "expert_ppi", "calibrated_ppi", "delta_calibrated"]
    ].rename(columns={"calibrated_ppi": "surrogate_ppi", "delta_calibrated": "delta"})


def compare_expert_vs_variants(
    match_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    bundle: PositionModelBundle,
) -> pd.DataFrame:
    expert_scored, _, _ = compute_ppi(match_df, weights_df, display_scale=bundle.display_scale)
    ols = bundle.batch_display_predictions(match_df, variant="ols")
    ridge = bundle.batch_display_predictions(match_df, variant="ridge")
    cal = _calibrated_display(match_df, bundle)

    out = match_df[["name", "profile"]].copy()
    out["expert_ppi"] = np.round(expert_scored["ppi"].values, 2)
    out["ols_ppi"] = np.round(ols, 2)
    out["ridge_ppi"] = np.round(ridge, 2)
    out["calibrated_ppi"] = np.round(cal, 2)
    out["delta_ols"] = out["ols_ppi"] - out["expert_ppi"]
    out["delta_ridge"] = out["ridge_ppi"] - out["expert_ppi"]
    out["delta_calibrated"] = out["calibrated_ppi"] - out["expert_ppi"]
    return out.sort_values("expert_ppi", ascending=False).reset_index(drop=True)
