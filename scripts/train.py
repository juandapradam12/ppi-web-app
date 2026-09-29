#!/usr/bin/env python3
"""Train position-specific linear surrogate models for PPI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppi.io import DATA_PROCESSED, DATA_RAW, MODELS_DIR, load_holdout_matches, load_training_frame
from ppi.models import train_position_models
from ppi.weights import load_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PROCESSED / "training_players.csv",
        help="Training CSV path",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DATA_RAW / "importance_weights.csv",
        help="Importance weights CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=MODELS_DIR,
        help="Output directory for scaler + models",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Regenerate synthetic training CSV before fitting",
    )
    args = parser.parse_args()

    if args.rebuild_data or not args.data.exists():
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "build_training_data.py")])

    train_df = load_training_frame(args.data)
    weights = load_weights(args.weights)
    holdouts = load_holdout_matches()
    bundle = train_position_models(train_df, weights, holdout_frames=holdouts)
    bundle.save(args.out)

    print(f"Saved models to {args.out}")
    for profile, metrics in bundle.metrics.items():
        if profile.startswith("_") or profile == "holdout":
            continue
        print(
            f"  {profile:12s} n={metrics.get('n_train', 0):.0f}  "
            f"R²={metrics.get('r2_train', float('nan')):.3f}  "
            f"MAE={metrics.get('mae_train', float('nan')):.3f}"
        )
    if "holdout" in bundle.metrics:
        print("Holdout (real matches, expert display PPI vs surrogate):")
        for mid, hm in bundle.metrics["holdout"].items():
            print(
                f"  {mid}: MAE={hm['mae']:.3f}  R²={hm['r2']:.3f}  "
                f"Spearman={hm['spearman']:.3f}  n={hm['n']:.0f}"
            )


if __name__ == "__main__":
    main()
