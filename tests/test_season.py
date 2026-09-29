"""Season aggregation tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppi.season import season_long_frame, season_ppi_frame
from ppi.weights import load_weights


def test_season_ppi_has_multi_match_players() -> None:
    weights = load_weights(ROOT / "data" / "raw" / "importance_weights.csv")
    season = season_ppi_frame(weights)
    assert len(season) >= 10
    assert "season_ppi" in season.columns
    multi = season[season["matches"] > 1]
    assert len(multi) >= 1


def test_season_long_two_matches() -> None:
    weights = load_weights(ROOT / "data" / "raw" / "importance_weights.csv")
    long = season_long_frame(weights)
    assert long["match_id"].nunique() >= 2
