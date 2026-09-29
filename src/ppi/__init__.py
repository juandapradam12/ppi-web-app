"""Player Performance Index (PPI) — position-aware soccer performance scoring."""

from .schema import FEATURE_COLUMNS, PROFILES, FeatureRow
from .scoring import compute_ppi, scale_features
from .explain import feature_contributions
from .ranking import rank_players
from .models import PositionModelBundle
from .validation import compare_expert_vs_surrogate, holdout_metrics

__all__ = [
    "FEATURE_COLUMNS",
    "PROFILES",
    "FeatureRow",
    "compute_ppi",
    "scale_features",
    "feature_contributions",
    "rank_players",
    "PositionModelBundle",
    "compare_expert_vs_surrogate",
    "holdout_metrics",
]

__version__ = "1.0.0"
