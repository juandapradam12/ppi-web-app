#!/usr/bin/env python3
"""Train position-specific linear surrogates + pairwise Model B ranker."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppi.io import DATA_PROCESSED, DATA_RAW, MODELS_DIR, load_holdout_matches, load_training_frame
from ppi.models import train_position_models
from ppi.ranking_model import train_pairwise_ranker
from ppi.weights import load_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PROCESSED / "training_players.csv")
    parser.add_argument("--weights", type=Path, default=DATA_RAW / "importance_weights.csv")
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    parser.add_argument("--rebuild-data", action="store_true")
    args = parser.parse_args()

    if args.rebuild_data or not args.data.exists():
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "build_training_data.py")])

    train_df = load_training_frame(args.data)
    weights = load_weights(args.weights)
    holdouts = load_holdout_matches()
    bundle = train_position_models(train_df, weights, holdout_frames=holdouts)
    bundle.save(args.out)

    ranker = train_pairwise_ranker(train_df, weights)
    ranker.save(args.out)

    print(f"Saved models to {args.out}")
    for profile, metrics in bundle.metrics.items():
        if profile.startswith("_") or profile in (
            "holdout",
            "holdout_variants",
            "leave_one_match_out",
        ):
            continue
        print(
            f"  {profile:12s} n={metrics.get('n_train', 0):.0f}  "
            f"Ridge R²={metrics.get('r2_ridge', float('nan')):.3f}  "
            f"α={metrics.get('ridge_alpha', float('nan')):.4f}"
        )
    if "holdout_variants" in bundle.metrics:
        print("Holdout pooled (in-sample calibrator fit):")
        for variant, hm in bundle.metrics["holdout_variants"].items():
            print(
                f"  {variant:12s} MAE={hm['mae']:.3f}  Spearman={hm['spearman']:.3f}"
            )
    if "leave_one_match_out" in bundle.metrics:
        pooled = bundle.metrics["leave_one_match_out"].get("pooled", {})
        print(
            "Leave-one-match-out calibrated: "
            f"MAE={pooled.get('mae', float('nan')):.3f}  "
            f"Spearman={pooled.get('spearman', float('nan')):.3f}  "
            f"n={pooled.get('n', 0):.0f}"
        )
    print("Model B (pairwise ranker) pair accuracy:")
    for profile in ranker.metrics.get("profiles", []):
        m = ranker.metrics[profile]
        print(f"  {profile:12s} acc={m['pair_accuracy']:.3f}  pairs={m['n_pairs']:.0f}")


if __name__ == "__main__":
    main()
