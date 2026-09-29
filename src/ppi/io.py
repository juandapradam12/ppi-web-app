"""I/O helpers for match CSVs and metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


def load_match_stats(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else DATA_RAW / "barca_el_clasico_2025-05-11.csv"
    df = pd.read_csv(path)
    required = {"name", "profile", *FEATURE_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Match CSV missing columns: {sorted(missing)}")
    return df


def load_match_meta(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DATA_RAW / "match_meta.json"
    return json.loads(path.read_text())


def load_training_frame(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else DATA_PROCESSED / "training_players.csv"
    return pd.read_csv(path)
