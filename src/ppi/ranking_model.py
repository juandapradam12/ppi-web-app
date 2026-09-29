"""Model B: pairwise ranking surrogate (interpretable logistic on feature diffs)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from .schema import FEATURE_COLUMNS, PROFILES
from .scoring import compute_ppi, scale_features


@dataclass
class PairwiseRanker:
    """
    Rank-first Model B: logistic regression on (x_i - x_j) predicting whether
    expert PPI_i > PPI_j within profile. Scores a player as sum of P(beat j).
    """

    scaler: MinMaxScaler
    models: dict[str, LogisticRegression]
    metrics: dict

    def score_squad(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """Return pairwise win-rate scores (higher = better) for a match cohort."""
        scaled, _ = scale_features(match_df, scaler=self.scaler, fit=False)
        rows = []
        for profile in match_df["profile"].unique():
            mask = match_df["profile"] == profile
            idxs = match_df.index[mask].tolist()
            if profile not in self.models or len(idxs) == 0:
                for i in idxs:
                    rows.append(
                        {
                            "name": match_df.loc[i, "name"],
                            "profile": profile,
                            "rank_score": 0.0,
                        }
                    )
                continue
            model = self.models[profile]
            X = scaled.loc[idxs, list(FEATURE_COLUMNS)].values
            names = match_df.loc[idxs, "name"].tolist()
            n = len(idxs)
            scores = np.zeros(n)
            if n == 1:
                scores[0] = 0.5
            else:
                for i in range(n):
                    wins = 0.0
                    for j in range(n):
                        if i == j:
                            continue
                        diff = (X[i] - X[j]).reshape(1, -1)
                        wins += float(model.predict_proba(diff)[0, 1])
                    scores[i] = wins / (n - 1)
            for name, sc in zip(names, scores):
                rows.append({"name": name, "profile": profile, "rank_score": sc})

        out = pd.DataFrame(rows)
        # Global rank by score within match (cross-profile comparable via score scale)
        out = out.sort_values("rank_score", ascending=False).reset_index(drop=True)
        out.insert(0, "rank_b", range(1, len(out) + 1))
        return out

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "rank_scaler.joblib")
        for profile, model in self.models.items():
            joblib.dump(model, path / f"ranker_{profile.lower()}.joblib")
        (path / "ranker_metrics.json").write_text(json.dumps(self.metrics, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "PairwiseRanker | None":
        path = Path(directory)
        if not (path / "ranker_metrics.json").exists():
            return None
        scaler = joblib.load(path / "rank_scaler.joblib")
        meta = json.loads((path / "ranker_metrics.json").read_text())
        models = {}
        for profile in meta.get("profiles", PROFILES):
            p = path / f"ranker_{profile.lower()}.joblib"
            if p.exists():
                models[profile] = joblib.load(p)
        return cls(scaler=scaler, models=models, metrics=meta)


def _pairwise_dataset(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator, max_pairs: int = 4000
) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j and not np.isclose(y[i], y[j])]
    if not pairs:
        return np.empty((0, X.shape[1])), np.empty(0)
    if len(pairs) > max_pairs:
        chosen = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in chosen]
    diffs = []
    labels = []
    for i, j in pairs:
        diffs.append(X[i] - X[j])
        labels.append(1 if y[i] > y[j] else 0)
    return np.asarray(diffs), np.asarray(labels)


def train_pairwise_ranker(
    train_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    *,
    random_state: int = 42,
) -> PairwiseRanker:
    scored, scaler, _ = compute_ppi(train_df, weights_df)
    scaled, _ = scale_features(scored, scaler=scaler, fit=False)
    rng = np.random.default_rng(random_state)
    models: dict[str, LogisticRegression] = {}
    metrics: dict = {"profiles": []}

    for profile in PROFILES:
        mask = scored["profile"] == profile
        if mask.sum() < 6:
            continue
        X = scaled.loc[mask, list(FEATURE_COLUMNS)].values
        y = scored.loc[mask, "ppi"].values
        diffs, labels = _pairwise_dataset(X, y, rng)
        if len(labels) < 20:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(diffs, labels)
        pred = clf.predict(diffs)
        models[profile] = clf
        metrics["profiles"].append(profile)
        metrics[profile] = {
            "n_players": float(mask.sum()),
            "n_pairs": float(len(labels)),
            "pair_accuracy": float(accuracy_score(labels, pred)),
        }

    return PairwiseRanker(scaler=scaler, models=models, metrics=metrics)


def compare_expert_vs_ranker(
    match_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    ranker: PairwiseRanker,
) -> pd.DataFrame:
    expert, _, _ = compute_ppi(match_df, weights_df)
    ranked_a = expert[["name", "profile", "ppi"]].copy()
    ranked_a = ranked_a.sort_values("ppi", ascending=False).reset_index(drop=True)
    ranked_a["rank_expert"] = range(1, len(ranked_a) + 1)
    ranked_b = ranker.score_squad(match_df)
    merged = ranked_a.merge(ranked_b, on=["name", "profile"], how="left")
    merged["rank_delta"] = merged["rank_b"] - merged["rank_expert"]
    return merged.sort_values("rank_expert").reset_index(drop=True)
