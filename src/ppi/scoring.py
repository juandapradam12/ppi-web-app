"""Min-max scaling and expert-weighted PPI computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .schema import FEATURE_COLUMNS
from .weights import weight_vector


def scale_features(
    features: pd.DataFrame,
    scaler: MinMaxScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Scale feature columns to [0, 1]. Optionally reuse a fitted scaler."""
    frame = features[list(FEATURE_COLUMNS)].astype(float).copy()
    if scaler is None:
        scaler = MinMaxScaler()
    values = scaler.fit_transform(frame.values) if fit else scaler.transform(frame.values)
    scaled = pd.DataFrame(values, columns=list(FEATURE_COLUMNS), index=features.index)
    return scaled, scaler


def raw_ppi_scores(scaled: pd.DataFrame, weights_df: pd.DataFrame, profiles: pd.Series) -> np.ndarray:
    """Dot product of position weights and scaled features (pre display rescale)."""
    scores = np.zeros(len(scaled), dtype=float)
    for idx, (profile, row) in enumerate(zip(profiles.tolist(), scaled.itertuples(index=False))):
        betas = np.asarray(weight_vector(weights_df, profile), dtype=float)
        x = np.asarray(row, dtype=float)
        scores[idx] = float(np.dot(betas, x))
    return scores


def compute_ppi(
    df: pd.DataFrame,
    weights_df: pd.DataFrame,
    *,
    scaler: MinMaxScaler | None = None,
    fit_scaler: bool = True,
    display_scale: float = 10.0,
) -> tuple[pd.DataFrame, MinMaxScaler, np.ndarray]:
    """
    Compute the Player Performance Index.

    Pipeline:
      1. MinMax-scale features across the provided cohort
      2. Score = w_profile · x_scaled
      3. Rescale raw scores to [0, display_scale] within the cohort

    Returns the input frame with ``ppi_raw`` / ``ppi`` columns, the scaler,
    and the cohort-level min-max parameters used on raw scores (for explain).
    """
    required = {"profile", *FEATURE_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing columns: {sorted(missing)}")

    out = df.copy()
    scaled, fitted_scaler = scale_features(out, scaler=scaler, fit=fit_scaler)
    raw = raw_ppi_scores(scaled, weights_df, out["profile"])

    raw_min, raw_max = float(raw.min()), float(raw.max())
    if np.isclose(raw_max, raw_min):
        display = np.full_like(raw, display_scale / 2.0)
    else:
        display = (raw - raw_min) / (raw_max - raw_min) * display_scale

    out["ppi_raw"] = raw
    out["ppi"] = display
    meta = np.array([raw_min, raw_max, display_scale], dtype=float)
    return out, fitted_scaler, meta


def display_scale_from_raw(
    raw: np.ndarray,
    *,
    display_scale: float = 10.0,
) -> np.ndarray:
    """Rescale raw PPI scores to [0, display_scale] within a cohort."""
    raw = np.asarray(raw, dtype=float)
    raw_min, raw_max = float(raw.min()), float(raw.max())
    if np.isclose(raw_max, raw_min):
        return np.full_like(raw, display_scale / 2.0)
    return (raw - raw_min) / (raw_max - raw_min) * display_scale
