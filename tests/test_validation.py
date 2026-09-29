"""Tests for holdout surrogate validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppi.io import load_holdout_matches, load_match_stats
from ppi.models import PositionModelBundle, train_position_models
from ppi.validation import compare_expert_vs_surrogate, holdout_metrics
from ppi.weights import load_weights


@pytest.fixture(scope="module")
def bundle():
    from ppi.io import load_training_frame

    train = load_training_frame()
    weights = load_weights(ROOT / "data" / "raw" / "importance_weights.csv")
    holdouts = load_holdout_matches()
    return train_position_models(train, weights, holdout_frames=holdouts)


def test_holdout_metrics_reasonable(bundle: PositionModelBundle) -> None:
    match = load_match_stats(match_id="el-clasico-2025-05-11")
    weights = load_weights(ROOT / "data" / "raw" / "importance_weights.csv")
    hm = holdout_metrics(match, weights, bundle)
    assert hm["n"] >= 10
    assert hm["mae"] < 2.0
    assert hm["spearman"] > 0.8


def test_compare_table(bundle: PositionModelBundle) -> None:
    match = load_match_stats(match_id="el-clasico-2025-05-11")
    weights = load_weights(ROOT / "data" / "raw" / "importance_weights.csv")
    df = compare_expert_vs_surrogate(match, weights, bundle)
    assert {"expert_ppi", "surrogate_ppi", "delta"}.issubset(df.columns)
