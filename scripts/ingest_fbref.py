#!/usr/bin/env python3
"""
Map FBref-style player rows into a PPI match CSV.

Usage:
  python scripts/ingest_fbref.py --template   # print column template + position map
  python scripts/ingest_fbref.py rows.json --out data/raw/my_match.csv --match-id demo-1

Each JSON row supports FBref-ish keys (see POSITION_MAP and FIELD_ALIASES).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppi.schema import FEATURE_COLUMNS

POSITION_MAP = {
    "FW": "Attacker",
    "FW,MF": "Attacker",
    "LW": "Attacker",
    "RW": "Attacker",
    "AM": "Midfielder",
    "DM": "Midfielder",
    "MF": "Midfielder",
    "CM": "Midfielder",
    "LB": "Defender",
    "RB": "Defender",
    "CB": "Defender",
    "DF": "Defender",
    "GK": "Goalkeeper",
}

FIELD_ALIASES = {
    "name": "name",
    "player": "name",
    "pos": "pos",
    "position": "pos",
    "min": "playing_time",
    "minutes": "playing_time",
    "gls": "goals",
    "goals": "goals",
    "ast": "assists",
    "assists": "assists",
    "kp": "key_passes",
    "key_passes": "key_passes",
    "cmp": "completed_passes",
    "completed_passes": "completed_passes",
    "cmp_pct": "success_rate",
    "success_rate": "success_rate",
    "touches": "touches",
    "sh": "shots",
    "shots": "shots",
    "sot": "shots_on_goal",
    "shots_on_goal": "shots_on_goal",
    "crs": "crosses",
    "crosses": "crosses",
    "mis": "turnovers_partial",
    "dis": "turnovers_partial2",
    "recov": "recoveries",
    "recoveries": "recoveries",
    "fls": "fouls",
    "fouls": "fouls",
    "crdy": "yellow_cards",
    "yellow_cards": "yellow_cards",
    "crdr": "red_cards",
    "red_cards": "red_cards",
    "sv": "saves",
    "saves": "saves",
    "cs": "clean_sheets",
    "offsides": "offsides",
    "outcome": "outcome",
}


def normalize_row(raw: dict, *, team_touches: float, default_outcome: float) -> dict:
    row = {k.lower(): v for k, v in raw.items()}
    mapped: dict = {}
    for src, dst in FIELD_ALIASES.items():
        if src in row and dst not in mapped:
            mapped[dst] = row[src]

    pos = str(mapped.pop("pos", raw.get("pos", "MF")))
    profile = POSITION_MAP.get(pos, "Midfielder")
    name = str(mapped.pop("name", raw.get("name", "Unknown")))

    turnovers = float(mapped.pop("turnovers_partial", 0) or 0) + float(
        mapped.pop("turnovers_partial2", 0) or 0
    )
    touches = float(mapped.get("touches", 0) or 0)
    poss = round(touches / team_touches * 100, 2) if team_touches else 0.0

    out = {col: 0.0 for col in FEATURE_COLUMNS}
    for col in FEATURE_COLUMNS:
        if col in mapped:
            out[col] = float(mapped[col])
    out["turnovers"] = turnovers if turnovers else out.get("turnovers", 0)
    out["possession_participation"] = poss
    out["outcome"] = float(mapped.get("outcome", default_outcome))
    out["name"] = name
    out["profile"] = profile
    return out


def print_template() -> None:
    print("PPI match CSV columns:")
    print("  name, profile, " + ", ".join(FEATURE_COLUMNS) + ", match_id")
    print("\nPosition map (FBref Pos -> PPI profile):")
    for k, v in sorted(POSITION_MAP.items()):
        print(f"  {k:8s} -> {v}")
    print("\nExample JSON row:")
    example = {
        "name": "Raphinha",
        "pos": "LW",
        "min": 90,
        "gls": 2,
        "ast": 0,
        "kp": 3,
        "cmp": 32,
        "cmp_pct": 66.7,
        "touches": 56,
        "sh": 6,
        "sot": 2,
        "crs": 8,
        "mis": 1,
        "dis": 1,
        "recov": 2,
        "fls": 1,
        "crdy": 0,
        "crdr": 0,
        "sv": 0,
        "offsides": 1,
    }
    print(json.dumps(example, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows_json", nargs="?", type=Path, help="JSON list of player rows")
    parser.add_argument("--out", type=Path, help="Output CSV path")
    parser.add_argument("--match-id", default="custom-match")
    parser.add_argument("--team-touches", type=float, default=600.0)
    parser.add_argument("--outcome", type=float, default=1.0, help="1=win, 0=loss")
    parser.add_argument("--template", action="store_true")
    args = parser.parse_args()

    if args.template or args.rows_json is None:
        print_template()
        return

    rows = json.loads(args.rows_json.read_text())
    if not isinstance(rows, list):
        raise SystemExit("rows JSON must be a list of objects")

    records = [
        normalize_row(r, team_touches=args.team_touches, default_outcome=args.outcome)
        for r in rows
    ]
    df = pd.DataFrame(records)
    df["match_id"] = args.match_id
    cols = ["name", "profile", *FEATURE_COLUMNS, "match_id"]
    df = df[cols]
    if not args.out:
        print(df.to_csv(index=False))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
