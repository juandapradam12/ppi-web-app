"""Feature-level contribution breakdown for interpretability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import FEATURE_COLUMNS, FEATURE_LABELS
from .weights import weight_vector


def feature_contributions(
    scaled_row: pd.Series | dict[str, float],
    profile: str,
    weights_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return per-feature signed contributions ``w_i * x_i`` for one scaled player.

    Contributions sum to the raw PPI (before cohort display rescale).
    """
    if isinstance(scaled_row, dict):
        values = [float(scaled_row[c]) for c in FEATURE_COLUMNS]
    else:
        values = [float(scaled_row[c]) for c in FEATURE_COLUMNS]

    weights = weight_vector(weights_df, profile)
    contrib = np.asarray(weights) * np.asarray(values)

    frame = pd.DataFrame(
        {
            "feature": list(FEATURE_COLUMNS),
            "label": [FEATURE_LABELS[c] for c in FEATURE_COLUMNS],
            "weight": weights,
            "scaled_value": values,
            "contribution": contrib,
        }
    )
    frame["abs_contribution"] = frame["contribution"].abs()
    return frame.sort_values("abs_contribution", ascending=False).reset_index(drop=True)


def top_drivers(contrib_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Largest-magnitude contribution rows."""
    return contrib_df.head(n).copy()
