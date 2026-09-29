"""Unit tests for the PPI core library."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppi.explain import feature_contributions
from ppi.io import load_match_stats
from ppi.models import train_position_models
from ppi.ranking import rank_players
from ppi.schema import FEATURE_COLUMNS, PROFILES
from ppi.scoring import compute_ppi
from ppi.weights import load_weights, weight_vector


@pytest.fixture
def match_df() -> pd.DataFrame:
    return load_match_stats()


@pytest.fixture
def weights_df() -> pd.DataFrame:
    return load_weights(ROOT / "data" / "raw" / "importance_weights.csv")


def test_match_csv_schema(match_df: pd.DataFrame) -> None:
    assert {"name", "profile", *FEATURE_COLUMNS}.issubset(match_df.columns)
    assert set(match_df["profile"]).issubset(set(PROFILES))
    assert len(match_df) >= 11


def test_weight_vectors_complete(weights_df: pd.DataFrame) -> None:
    for profile in PROFILES:
        w = weight_vector(weights_df, profile)
        assert len(w) == len(FEATURE_COLUMNS)
        assert any(abs(v) > 0 for v in w)


def test_compute_ppi_range(match_df: pd.DataFrame, weights_df: pd.DataFrame) -> None:
    scored, scaler, meta = compute_ppi(match_df, weights_df)
    assert "ppi" in scored.columns
    assert scored["ppi"].min() >= 0
    assert scored["ppi"].max() <= 10 + 1e-9
    assert scaler is not None
    assert meta.shape == (3,)


def test_contributions_sum_to_raw(match_df: pd.DataFrame, weights_df: pd.DataFrame) -> None:
    scored, scaler, _ = compute_ppi(match_df, weights_df)
    from ppi.scoring import scale_features

    scaled, _ = scale_features(scored, scaler=scaler, fit=False)
    row = scored.iloc[0]
    contrib = feature_contributions(scaled.iloc[0], row["profile"], weights_df)
    assert abs(contrib["contribution"].sum() - row["ppi_raw"]) < 1e-6


def test_ranking_order(match_df: pd.DataFrame, weights_df: pd.DataFrame) -> None:
    scored, _, _ = compute_ppi(match_df, weights_df)
    ranked = rank_players(scored, min_minutes=30)
    assert ranked["rank"].iloc[0] == 1
    assert ranked["ppi"].is_monotonic_decreasing
    assert (ranked["playing_time"] >= 30).all()


def test_train_models_smoke(match_df: pd.DataFrame, weights_df: pd.DataFrame) -> None:
    # Expand match rows slightly so each profile has enough samples
    frames = [match_df]
    for i in range(5):
        noisy = match_df.copy()
        noisy["name"] = noisy["name"] + f"_{i}"
        noisy["touches"] = noisy["touches"] * (1 + 0.05 * i)
        frames.append(noisy)
    train = pd.concat(frames, ignore_index=True)
    bundle = train_position_models(train, weights_df)
    assert set(bundle.models).issubset(set(PROFILES))
    pred = bundle.predict_row("Attacker", {c: float(match_df.iloc[0][c]) for c in FEATURE_COLUMNS})
    assert 0 <= pred <= 15  # soft bound; display scale is 10 but LR may extrapolate slightly
