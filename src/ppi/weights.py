"""Position-specific expert importance weights."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import FEATURE_COLUMNS, PROFILES, validate_profiles

# Domain priors: positive contribution for productive actions, negative for
# disruptive ones. Magnitude reflects role emphasis (attackers score on goals /
# shots; defenders on recoveries / clean sheets; keepers on saves).
DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "Attacker": {
        "playing_time": 0.06,
        "goals": 0.28,
        "assists": 0.16,
        "key_passes": 0.10,
        "completed_passes": 0.03,
        "success_rate": 0.04,
        "possession_participation": 0.05,
        "touches": 0.03,
        "shots": 0.08,
        "shots_on_goal": 0.10,
        "crosses": 0.04,
        "turnovers": -0.06,
        "recoveries": 0.02,
        "fouls": -0.03,
        "yellow_cards": -0.04,
        "red_cards": -0.08,
        "saves": 0.00,
        "clean_sheets": 0.00,
        "offsides": -0.03,
        "outcome": 0.11,
    },
    "Midfielder": {
        "playing_time": 0.07,
        "goals": 0.10,
        "assists": 0.12,
        "key_passes": 0.14,
        "completed_passes": 0.10,
        "success_rate": 0.09,
        "possession_participation": 0.08,
        "touches": 0.06,
        "shots": 0.04,
        "shots_on_goal": 0.04,
        "crosses": 0.03,
        "turnovers": -0.07,
        "recoveries": 0.08,
        "fouls": -0.04,
        "yellow_cards": -0.05,
        "red_cards": -0.08,
        "saves": 0.00,
        "clean_sheets": 0.00,
        "offsides": -0.02,
        "outcome": 0.11,
    },
    "Defender": {
        "playing_time": 0.08,
        "goals": 0.06,
        "assists": 0.05,
        "key_passes": 0.04,
        "completed_passes": 0.08,
        "success_rate": 0.07,
        "possession_participation": 0.05,
        "touches": 0.04,
        "shots": 0.02,
        "shots_on_goal": 0.02,
        "crosses": 0.03,
        "turnovers": -0.05,
        "recoveries": 0.14,
        "fouls": -0.06,
        "yellow_cards": -0.07,
        "red_cards": -0.10,
        "saves": 0.00,
        "clean_sheets": 0.12,
        "offsides": -0.01,
        "outcome": 0.12,
    },
    "Goalkeeper": {
        "playing_time": 0.10,
        "goals": 0.00,
        "assists": 0.01,
        "key_passes": 0.01,
        "completed_passes": 0.06,
        "success_rate": 0.07,
        "possession_participation": 0.03,
        "touches": 0.03,
        "shots": 0.00,
        "shots_on_goal": 0.00,
        "crosses": 0.00,
        "turnovers": -0.04,
        "recoveries": 0.04,
        "fouls": -0.03,
        "yellow_cards": -0.05,
        "red_cards": -0.10,
        "saves": 0.28,
        "clean_sheets": 0.18,
        "offsides": 0.00,
        "outcome": 0.14,
    },
}


def weights_frame(weights: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """Return a tidy DataFrame of profile × feature weights."""
    source = weights or DEFAULT_WEIGHTS
    validate_profiles(source.keys())
    rows = []
    for profile in PROFILES:
        row = {"profile": profile}
        for col in FEATURE_COLUMNS:
            row[col] = float(source[profile][col])
        rows.append(row)
    return pd.DataFrame(rows)


def load_weights(path: str | Path | None = None) -> pd.DataFrame:
    """Load weights CSV or fall back to built-in defaults."""
    if path is None:
        return weights_frame()
    path = Path(path)
    if not path.exists():
        return weights_frame()
    df = pd.read_csv(path)
    if "profile" not in df.columns:
        raise ValueError("Weights CSV must include a 'profile' column")
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Weights CSV missing columns: {missing}")
    validate_profiles(df["profile"].tolist())
    return df[["profile", *FEATURE_COLUMNS]].copy()


def weight_vector(weights_df: pd.DataFrame, profile: str) -> list[float]:
    row = weights_df.loc[weights_df["profile"] == profile]
    if row.empty:
        raise ValueError(f"No weights for profile {profile!r}")
    return [float(row.iloc[0][col]) for col in FEATURE_COLUMNS]
