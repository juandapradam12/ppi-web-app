"""Linear-regression surrogate models per playing profile."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .schema import FEATURE_COLUMNS, PROFILES
from .scoring import compute_ppi, scale_features


@dataclass
class PositionModelBundle:
    """Fitted MinMax scaler + one LinearRegression per profile (targets: expert PPI raw)."""

    scaler: MinMaxScaler
    models: dict[str, LinearRegression]
    metrics: dict[str, dict]
    display_scale: float = 10.0

    def predict_raw(self, profile: str, features: dict[str, float] | pd.Series) -> float:
        if profile not in self.models:
            raise ValueError(f"No model for profile {profile!r}")
        if isinstance(features, pd.Series):
            vector = np.array([[float(features[c]) for c in FEATURE_COLUMNS]], dtype=float)
        else:
            vector = np.array([[float(features[c]) for c in FEATURE_COLUMNS]], dtype=float)
        scaled = self.scaler.transform(vector)
        return float(self.models[profile].predict(scaled)[0])

    def predict_row(self, profile: str, features: dict[str, float] | pd.Series) -> float:
        """Display-scale approximation using training raw quantiles."""
        raw = self.predict_raw(profile, features)
        bounds = self.metrics.get("_raw_bounds", {})
        lo = float(bounds.get("min", raw))
        hi = float(bounds.get("max", raw))
        if np.isclose(hi, lo):
            return round(self.display_scale / 2.0, 2)
        display = (raw - lo) / (hi - lo) * self.display_scale
        return round(float(np.clip(display, 0, self.display_scale)), 2)

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "scaler.joblib")
        for profile, model in self.models.items():
            joblib.dump(model, path / f"model_{profile.lower()}.joblib")
        payload = {
            "metrics": self.metrics,
            "display_scale": self.display_scale,
            "profiles": list(self.models),
        }
        (path / "metrics.json").write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "PositionModelBundle":
        path = Path(directory)
        scaler = joblib.load(path / "scaler.joblib")
        meta = json.loads((path / "metrics.json").read_text())
        models = {}
        for profile in meta.get("profiles", PROFILES):
            models[profile] = joblib.load(path / f"model_{profile.lower()}.joblib")
        return cls(
            scaler=scaler,
            models=models,
            metrics=meta.get("metrics", {}),
            display_scale=float(meta.get("display_scale", 10.0)),
        )


def train_position_models(
    train_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    *,
    holdout_frames: list[pd.DataFrame] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    display_scale: float = 10.0,
) -> PositionModelBundle:
    """
    Train surrogates on synthetic data only.

    Targets are expert ``ppi_raw`` (not cohort-rescaled display PPI) so holdout
    real matches can be evaluated honestly.
    """
    scored, scaler, _ = compute_ppi(train_df, weights_df, display_scale=display_scale)
    scaled, _ = scale_features(scored, scaler=scaler, fit=False)
    y_all = scored["ppi_raw"].values

    models: dict[str, LinearRegression] = {}
    metrics: dict[str, dict] = {
        "_raw_bounds": {
            "min": float(y_all.min()),
            "max": float(y_all.max()),
        }
    }

    for profile in PROFILES:
        mask = scored["profile"] == profile
        if mask.sum() == 0:
            continue

        X = scaled.loc[mask, list(FEATURE_COLUMNS)].values
        y = scored.loc[mask, "ppi_raw"].values

        if len(y) < 8:
            lr = LinearRegression().fit(X, y)
            pred = lr.predict(X)
            models[profile] = lr
            metrics[profile] = {
                "n_train": float(len(y)),
                "r2_train": float(r2_score(y, pred)),
                "mae_train": float(mean_absolute_error(y, pred)),
            }
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        lr = LinearRegression().fit(X_train, y_train)
        pred = lr.predict(X_test)
        models[profile] = lr
        metrics[profile] = {
            "n_train": float(len(y)),
            "r2_train": float(r2_score(y_test, pred)),
            "mae_train": float(mean_absolute_error(y_test, pred)),
        }

    bundle = PositionModelBundle(
        scaler=scaler, models=models, metrics=metrics, display_scale=display_scale
    )

    if holdout_frames:
        from .validation import holdout_metrics

        metrics["holdout"] = {}
        for frame in holdout_frames:
            match_id = str(frame["match_id"].iloc[0]) if "match_id" in frame.columns else "holdout"
            metrics["holdout"][match_id] = holdout_metrics(frame, weights_df, bundle)

    return bundle
