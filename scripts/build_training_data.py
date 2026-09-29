#!/usr/bin/env python3
"""Build synthetic training corpus; real FBref matches stay holdout-only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppi.io import DATA_PROCESSED, DATA_RAW
from ppi.schema import FEATURE_COLUMNS

HOLDOUT_IDS = {m["id"] for m in json.loads((DATA_RAW / "matches.json").read_text())}


def _clip_row(d: dict) -> dict:
    d["playing_time"] = float(np.clip(d["playing_time"], 1, 90))
    d["goals"] = float(np.clip(d["goals"], 0, 4))
    d["assists"] = float(np.clip(d["assists"], 0, 4))
    d["key_passes"] = float(np.clip(d["key_passes"], 0, 12))
    d["completed_passes"] = float(np.clip(d["completed_passes"], 0, 120))
    d["success_rate"] = float(np.clip(d["success_rate"], 40, 100))
    d["possession_participation"] = float(np.clip(d["possession_participation"], 0.5, 20))
    d["touches"] = float(np.clip(d["touches"], 5, 120))
    d["shots"] = float(np.clip(d["shots"], 0, 10))
    d["shots_on_goal"] = float(np.clip(min(d["shots_on_goal"], d["shots"]), 0, 8))
    d["crosses"] = float(np.clip(d["crosses"], 0, 12))
    d["turnovers"] = float(np.clip(d["turnovers"], 0, 12))
    d["recoveries"] = float(np.clip(d["recoveries"], 0, 15))
    d["fouls"] = float(np.clip(d["fouls"], 0, 6))
    d["yellow_cards"] = float(np.clip(d["yellow_cards"], 0, 2))
    d["red_cards"] = float(np.clip(d["red_cards"], 0, 1))
    d["saves"] = float(np.clip(d["saves"], 0, 10))
    d["clean_sheets"] = float(np.clip(d["clean_sheets"], 0, 1))
    d["offsides"] = float(np.clip(d["offsides"], 0, 4))
    d["outcome"] = float(1 if d["outcome"] >= 0.5 else 0)
    return d


def load_seed_matches() -> pd.DataFrame:
    """Real matches used only as perturbation seeds (rows tagged with holdout ids are excluded from output)."""
    frames = []
    for entry in json.loads((DATA_RAW / "matches.json").read_text()):
        path = DATA_RAW / entry["csv"]
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_training_frame(rng: np.random.Generator | None = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(42)
    cols = ["name", "profile", *FEATURE_COLUMNS, "match_id"]
    seeds = load_seed_matches()
    synth_rows: list[dict] = []

    for _, row in seeds.iterrows():
        base = row.to_dict()
        for k in range(22):
            d = dict(base)
            for col in FEATURE_COLUMNS:
                if col in ("outcome", "yellow_cards", "red_cards", "clean_sheets"):
                    d[col] = float(base[col])
                else:
                    d[col] = float(base[col]) * float(1 + rng.normal(0, 0.2))
            if k % 4 == 0:
                d["outcome"] = float(rng.integers(0, 2))
            d = _clip_row(d)
            d["name"] = f"{base['name']}#sim{k}"
            d["match_id"] = f"synthetic-{k}"
            synth_rows.append(d)

    archetypes = {
        "Attacker": dict(
            playing_time=85, goals=1.2, assists=0.6, key_passes=2.5, completed_passes=28,
            success_rate=78, possession_participation=8, touches=50, shots=4, shots_on_goal=1.5,
            crosses=2, turnovers=3, recoveries=2, fouls=1, yellow_cards=0, red_cards=0, saves=0,
            clean_sheets=0, offsides=0.5, outcome=1,
        ),
        "Midfielder": dict(
            playing_time=88, goals=0.3, assists=0.5, key_passes=2.0, completed_passes=65,
            success_rate=90, possession_participation=12, touches=80, shots=1, shots_on_goal=0.3,
            crosses=1, turnovers=2, recoveries=5, fouls=1, yellow_cards=0.2, red_cards=0, saves=0,
            clean_sheets=0, offsides=0, outcome=1,
        ),
        "Defender": dict(
            playing_time=90, goals=0.1, assists=0.1, key_passes=0.3, completed_passes=55,
            success_rate=88, possession_participation=9, touches=60, shots=0.4, shots_on_goal=0.1,
            crosses=0.5, turnovers=1, recoveries=6, fouls=1.2, yellow_cards=0.3, red_cards=0,
            saves=0, clean_sheets=0.4, offsides=0, outcome=1,
        ),
        "Goalkeeper": dict(
            playing_time=90, goals=0, assists=0, key_passes=0, completed_passes=25,
            success_rate=82, possession_participation=4, touches=30, shots=0, shots_on_goal=0,
            crosses=0, turnovers=0.2, recoveries=1, fouls=0, yellow_cards=0, red_cards=0,
            saves=3.5, clean_sheets=0.45, offsides=0, outcome=1,
        ),
    }

    for profile, arch in archetypes.items():
        for i in range(55):
            d = {c: float(arch[c]) * float(1 + rng.normal(0, 0.25)) for c in FEATURE_COLUMNS}
            d["outcome"] = float(rng.integers(0, 2))
            d = _clip_row(d)
            if profile != "Goalkeeper":
                d["saves"] = 0.0
                d["clean_sheets"] = float(rng.integers(0, 2)) if profile == "Defender" else 0.0
            d["name"] = f"{profile[:3].upper()}-synth-{i}"
            d["profile"] = profile
            d["match_id"] = f"arch-{profile}-{i}"
            synth_rows.append(d)

    train_df = pd.DataFrame(synth_rows)[cols]
    train_df = train_df[~train_df["match_id"].isin(HOLDOUT_IDS)]
    return train_df


def main() -> None:
    out = DATA_PROCESSED / "training_players.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = build_training_frame()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} synthetic rows to {out}")
    print(df["profile"].value_counts().to_string())


if __name__ == "__main__":
    main()
