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
MATCHES_INDEX = DATA_RAW / "matches.json"


def list_matches() -> list[dict]:
    if not MATCHES_INDEX.exists():
        return []
    return json.loads(MATCHES_INDEX.read_text())


def match_entry(match_id: str) -> dict:
    for entry in list_matches():
        if entry["id"] == match_id:
            return entry
    raise ValueError(f"Unknown match_id {match_id!r}")


def load_match_stats(match_id: str | None = None, path: str | Path | None = None) -> pd.DataFrame:
    if path is not None:
        csv_path = Path(path)
    else:
        entries = list_matches()
        if not entries:
            csv_path = DATA_RAW / "barca_el_clasico_2025-05-11.csv"
        else:
            mid = match_id or entries[0]["id"]
            csv_path = DATA_RAW / match_entry(mid)["csv"]
    df = pd.read_csv(csv_path)
    required = {"name", "profile", *FEATURE_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Match CSV missing columns: {sorted(missing)}")
    return df


def load_match_meta(match_id: str | None = None, path: str | Path | None = None) -> dict:
    if path is not None:
        meta_path = Path(path)
    else:
        entries = list_matches()
        mid = match_id or (entries[0]["id"] if entries else None)
        if mid is None:
            meta_path = DATA_RAW / "match_meta.json"
        else:
            meta_path = DATA_RAW / match_entry(mid)["meta"]
    return json.loads(meta_path.read_text())


def load_holdout_matches() -> list[pd.DataFrame]:
    frames = []
    for entry in list_matches():
        if entry.get("holdout"):
            frames.append(load_match_stats(match_id=entry["id"]))
    return frames


def load_training_frame(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else DATA_PROCESSED / "training_players.csv"
    return pd.read_csv(path)
