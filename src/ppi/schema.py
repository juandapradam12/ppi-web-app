"""Canonical feature schema for the Player Performance Index."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

PROFILES = ("Attacker", "Midfielder", "Defender", "Goalkeeper")

# Ordered feature vector used by scoring, scaling, and models.
FEATURE_COLUMNS: tuple[str, ...] = (
    "playing_time",
    "goals",
    "assists",
    "key_passes",
    "completed_passes",
    "success_rate",
    "possession_participation",
    "touches",
    "shots",
    "shots_on_goal",
    "crosses",
    "turnovers",
    "recoveries",
    "fouls",
    "yellow_cards",
    "red_cards",
    "saves",
    "clean_sheets",
    "offsides",
    "outcome",
)

# Human-readable labels for dashboards / contribution charts.
FEATURE_LABELS: dict[str, str] = {
    "playing_time": "Playing time (min)",
    "goals": "Goals",
    "assists": "Assists",
    "key_passes": "Key passes",
    "completed_passes": "Completed passes",
    "success_rate": "Pass completion %",
    "possession_participation": "Possession share %",
    "touches": "Touches",
    "shots": "Shots",
    "shots_on_goal": "Shots on target",
    "crosses": "Crosses",
    "turnovers": "Turnovers",
    "recoveries": "Recoveries",
    "fouls": "Fouls",
    "yellow_cards": "Yellow cards",
    "red_cards": "Red cards",
    "saves": "Saves",
    "clean_sheets": "Clean sheet",
    "offsides": "Offsides",
    "outcome": "Match win",
}

# Features where higher is worse — weights are typically negative.
NEGATIVE_FEATURES: frozenset[str] = frozenset(
    {"turnovers", "fouls", "yellow_cards", "red_cards", "offsides"}
)


@dataclass(frozen=True)
class FeatureRow:
    """One player-match observation in PPI feature space."""

    name: str
    profile: str
    playing_time: float
    goals: float
    assists: float
    key_passes: float
    completed_passes: float
    success_rate: float
    possession_participation: float
    touches: float
    shots: float
    shots_on_goal: float
    crosses: float
    turnovers: float
    recoveries: float
    fouls: float
    yellow_cards: float
    red_cards: float
    saves: float
    clean_sheets: float
    offsides: float
    outcome: float
    match_id: str = ""

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError(f"Unknown profile {self.profile!r}; expected one of {PROFILES}")

    def feature_dict(self) -> dict[str, float]:
        data = asdict(self)
        return {col: float(data[col]) for col in FEATURE_COLUMNS}

    def feature_vector(self) -> list[float]:
        return [self.feature_dict()[col] for col in FEATURE_COLUMNS]


def validate_profiles(profiles: Iterable[str]) -> None:
    unknown = sorted({p for p in profiles if p not in PROFILES})
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}")
