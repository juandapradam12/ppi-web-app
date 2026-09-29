"""Uncertainty and Model B tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppi.io import load_holdout_matches, load_match_stats, load_training_frame
from ppi.models import train_position_models
from ppi.ranking_model import compare_expert_vs_ranker, train_pairwise_ranker
from ppi.uncertainty import bootstrap_ppi_intervals
from ppi.weights import load_weights


def test_bootstrap_intervals() -> None:
    match = load_match_stats(match_id="el-clasico-2025-05-11")
    weights = load_weights(ROOT / "data" / "raw" / "importance_weights.csv")
    out = bootstrap_ppi_intervals(match, weights, n_boot=50, min_minutes=30)
    assert len(out) >= 5
    assert (out["ppi_hi"] >= out["ppi_lo"]).all()
    assert (out["top3_prob"] >= 0).all() and (out["top3_prob"] <= 1).all()


def test_pairwise_ranker_smoke() -> None:
    train = load_training_frame()
    weights = load_weights(ROOT / "data" / "raw" / "importance_weights.csv")
    ranker = train_pairwise_ranker(train.sample(n=min(200, len(train)), random_state=0), weights)
    assert len(ranker.models) >= 1
    match = load_match_stats(match_id="celta-2025-04-19")
    cmp = compare_expert_vs_ranker(match, weights, ranker)
    assert "rank_b" in cmp.columns


def test_three_holdout_matches() -> None:
    frames = load_holdout_matches()
    assert len(frames) >= 3


def test_leave_one_match_out_present() -> None:
    train = load_training_frame()
    weights = load_weights(ROOT / "data" / "raw" / "importance_weights.csv")
    holdouts = load_holdout_matches()
    # subsample train for speed
    bundle = train_position_models(
        train.sample(n=min(300, len(train)), random_state=1),
        weights,
        holdout_frames=holdouts,
    )
    assert "leave_one_match_out" in bundle.metrics
    assert "pooled" in bundle.metrics["leave_one_match_out"]
    assert bundle.metrics["leave_one_match_out"]["pooled"]["n"] >= 30
