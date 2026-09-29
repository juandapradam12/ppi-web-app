#!/usr/bin/env python3
"""Score a match CSV with the expert PPI and write ranked output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppi.io import DATA_PROCESSED, DATA_RAW, load_match_stats
from ppi.ranking import rank_players
from ppi.scoring import compute_ppi
from ppi.weights import load_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", type=Path, default=DATA_RAW / "barca_el_clasico_2025-05-11.csv")
    parser.add_argument("--weights", type=Path, default=DATA_RAW / "importance_weights.csv")
    parser.add_argument("--min-minutes", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=DATA_PROCESSED / "el_clasico_ranked.csv")
    args = parser.parse_args()

    match = load_match_stats(args.match)
    weights = load_weights(args.weights)
    scored, _, _ = compute_ppi(match, weights)
    ranked = rank_players(scored, min_minutes=args.min_minutes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.out, index=False)
    print(ranked.to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
