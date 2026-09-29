"""Surrogate models: OLS baseline, RidgeCV, and isotonic calibration on holdout."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .schema import FEATURE_COLUMNS, PROFILES
from .scoring import compute_ppi, display_scale_from_raw, scale_features

Variant = Literal["ols", "ridge", "calibrated"]


@dataclass
class PositionModelBundle:
    """
    Fitted MinMax scaler + per-profile regressors (Ridge primary, OLS for comparison)
    + optional isotonic calibrator fit on pooled holdout squads.
    """

    scaler: MinMaxScaler
    models: dict[str, RidgeCV | LinearRegression]
    metrics: dict[str, Any]
    display_scale: float = 10.0
    ols_models: dict[str, LinearRegression] = field(default_factory=dict)
    calibrator: IsotonicRegression | None = None

    def _vector(self, features: dict[str, float] | pd.Series) -> np.ndarray:
        if isinstance(features, pd.Series):
            return np.array([[float(features[c]) for c in FEATURE_COLUMNS]], dtype=float)
        return np.array([[float(features[c]) for c in FEATURE_COLUMNS]], dtype=float)

    def predict_raw(
        self,
        profile: str,
        features: dict[str, float] | pd.Series,
        *,
        variant: Variant = "ridge",
    ) -> float:
        if variant == "ols":
            model = self.ols_models.get(profile)
        else:
            model = self.models.get(profile)
        if model is None:
            raise ValueError(f"No model for profile {profile!r} variant={variant}")
        scaled = self.scaler.transform(self._vector(features))
        return float(model.predict(scaled)[0])

    def _raw_to_training_display(self, raw: float) -> float:
        bounds = self.metrics.get("_raw_bounds", {})
        lo = float(bounds.get("min", raw))
        hi = float(bounds.get("max", raw))
        if np.isclose(hi, lo):
            return self.display_scale / 2.0
        return float(np.clip((raw - lo) / (hi - lo) * self.display_scale, 0, self.display_scale))

    def predict_display(
        self,
        profile: str,
        features: dict[str, float] | pd.Series,
        *,
        variant: Variant = "calibrated",
    ) -> float:
        """Display PPI (0–10) for a single what-if row (training-scale Ridge/OLS + optional isotonic)."""
        base: Variant = "ridge" if variant == "calibrated" else variant
        raw = self.predict_raw(profile, features, variant=base)
        display_val = self._raw_to_training_display(raw)
        if variant == "calibrated" and self.calibrator is not None:
            return round(float(self.calibrator.predict([display_val])[0]), 2)
        return round(display_val, 2)

    def predict_row(self, profile: str, features: dict[str, float] | pd.Series) -> float:
        """Default production path: Ridge display + isotonic calibration."""
        return self.predict_display(profile, features, variant="calibrated")

    def batch_display_predictions(
        self,
        match_df: pd.DataFrame,
        *,
        variant: Variant = "ridge",
    ) -> np.ndarray:
        """Within-match cohort display PPI for a variant (before isotonic)."""
        raw_preds = np.array(
            [self.predict_raw(str(r["profile"]), r, variant=variant) for _, r in match_df.iterrows()],
            dtype=float,
        )
        return display_scale_from_raw(raw_preds, display_scale=self.display_scale)

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "scaler.joblib")
        for profile, model in self.models.items():
            joblib.dump(model, path / f"model_{profile.lower()}.joblib")
        for profile, model in self.ols_models.items():
            joblib.dump(model, path / f"ols_{profile.lower()}.joblib")
        if self.calibrator is not None:
            joblib.dump(self.calibrator, path / "calibrator.joblib")
        payload = {
            "metrics": self.metrics,
            "display_scale": self.display_scale,
            "profiles": list(self.models),
            "has_calibrator": self.calibrator is not None,
        }
        (path / "metrics.json").write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "PositionModelBundle":
        path = Path(directory)
        scaler = joblib.load(path / "scaler.joblib")
        meta = json.loads((path / "metrics.json").read_text())
        models = {}
        ols_models = {}
        for profile in meta.get("profiles", PROFILES):
            models[profile] = joblib.load(path / f"model_{profile.lower()}.joblib")
            ols_path = path / f"ols_{profile.lower()}.joblib"
            if ols_path.exists():
                ols_models[profile] = joblib.load(ols_path)
        calibrator = None
        cal_path = path / "calibrator.joblib"
        if cal_path.exists():
            calibrator = joblib.load(cal_path)
        return cls(
            scaler=scaler,
            models=models,
            ols_models=ols_models,
            calibrator=calibrator,
            metrics=meta.get("metrics", {}),
            display_scale=float(meta.get("display_scale", 10.0)),
        )


def _fit_profile_models(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[LinearRegression, RidgeCV, dict[str, float]]:
    ols = LinearRegression()
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 30), cv=min(5, max(2, len(y) // 4)))

    if len(y) < 8:
        ols.fit(X, y)
        ridge.fit(X, y)
        pred_ols = ols.predict(X)
        pred_ridge = ridge.predict(X)
        stats = {
            "r2_ols": float(r2_score(y, pred_ols)),
            "mae_ols": float(mean_absolute_error(y, pred_ols)),
            "r2_ridge": float(r2_score(y, pred_ridge)),
            "mae_ridge": float(mean_absolute_error(y, pred_ridge)),
            "ridge_alpha": float(ridge.alpha_) if hasattr(ridge, "alpha_") else float("nan"),
        }
        return ols, ridge, stats

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    ols.fit(X_train, y_train)
    ridge.fit(X_train, y_train)
    stats = {
        "r2_ols": float(r2_score(y_test, ols.predict(X_test))),
        "mae_ols": float(mean_absolute_error(y_test, ols.predict(X_test))),
        "r2_ridge": float(r2_score(y_test, ridge.predict(X_test))),
        "mae_ridge": float(mean_absolute_error(y_test, ridge.predict(X_test))),
        "ridge_alpha": float(ridge.alpha_),
    }
    return ols, ridge, stats


def _fit_isotonic_calibrator(
    holdout_frames: list[pd.DataFrame],
    weights_df: pd.DataFrame,
    bundle: PositionModelBundle,
) -> IsotonicRegression | None:
    xs: list[float] = []
    ys: list[float] = []
    for frame in holdout_frames:
        expert, _, _ = compute_ppi(frame, weights_df, display_scale=bundle.display_scale)
        ridge_display = bundle.batch_display_predictions(frame, variant="ridge")
        xs.extend(ridge_display.tolist())
        ys.extend(expert["ppi"].tolist())
    if len(xs) < 5:
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.array(xs), np.array(ys))
    return iso


def train_position_models(
    train_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    *,
    holdout_frames: list[pd.DataFrame] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    display_scale: float = 10.0,
) -> PositionModelBundle:
    """Train OLS + Ridge on synthetic ``ppi_raw``; calibrate Ridge on holdout squads."""
    scored, scaler, _ = compute_ppi(train_df, weights_df, display_scale=display_scale)
    scaled, _ = scale_features(scored, scaler=scaler, fit=False)
    y_all = scored["ppi_raw"].values

    ridge_models: dict[str, RidgeCV] = {}
    ols_models: dict[str, LinearRegression] = {}
    metrics: dict[str, Any] = {
        "_raw_bounds": {"min": float(y_all.min()), "max": float(y_all.max())},
    }

    for profile in PROFILES:
        mask = scored["profile"] == profile
        if mask.sum() == 0:
            continue
        X = scaled.loc[mask, list(FEATURE_COLUMNS)].values
        y = scored.loc[mask, "ppi_raw"].values
        ols, ridge, stats = _fit_profile_models(
            X, y, test_size=test_size, random_state=random_state
        )
        ols_models[profile] = ols
        ridge_models[profile] = ridge
        metrics[profile] = {"n_train": float(len(y)), **stats}

    bundle = PositionModelBundle(
        scaler=scaler,
        models=ridge_models,
        ols_models=ols_models,
        metrics=metrics,
        display_scale=display_scale,
    )

    if holdout_frames:
        from .validation import (
            holdout_metrics,
            holdout_metrics_all_variants,
            leave_one_match_out_metrics,
        )

        # Honest evaluation: calibrate on other matches, score the left-out match
        lmo = leave_one_match_out_metrics(holdout_frames, weights_df, bundle)
        metrics["leave_one_match_out"] = lmo

        # Production calibrator fit on all demo holdouts (documented in UI/README)
        bundle.calibrator = _fit_isotonic_calibrator(holdout_frames, weights_df, bundle)
        metrics["holdout_variants"] = holdout_metrics_all_variants(
            holdout_frames, weights_df, bundle
        )
        metrics["holdout"] = {
            str(frame["match_id"].iloc[0]): holdout_metrics(
                frame, weights_df, bundle, variant="calibrated"
            )
            for frame in holdout_frames
        }
        # Conformal residual scale from LMO calibrated errors
        if lmo.get("pooled_residuals"):
            res = np.asarray(lmo["pooled_residuals"], dtype=float)
            metrics["_conformal"] = {
                "q90": float(np.quantile(np.abs(res), 0.90, method="higher")),
                "q80": float(np.quantile(np.abs(res), 0.80, method="higher")),
                "n": float(len(res)),
            }
            # drop bulky residuals from persisted metrics
            metrics["leave_one_match_out"] = {
                k: v for k, v in lmo.items() if k != "pooled_residuals"
            }

    return bundle
