#!/usr/bin/env python3
"""Train position-specific linear surrogate models for PPI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppi.io import DATA_PROCESSED, DATA_RAW, MODELS_DIR, load_training_frame
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
    args = parser.parse_args()

    df = load_training_frame(args.data)
    weights = load_weights(args.weights)
    bundle = train_position_models(df, weights)
    bundle.save(args.out)

    print(f"Saved models to {args.out}")
    for profile, metrics in bundle.metrics.items():
        print(
            f"  {profile:12s} n={metrics.get('n', 0):.0f}  "
            f"R²={metrics.get('r2', float('nan')):.3f}  "
            f"MAE={metrics.get('mae', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
