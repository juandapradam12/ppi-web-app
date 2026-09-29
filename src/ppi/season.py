"""Minutes-weighted season PPI across multiple match CSVs."""

from __future__ import annotations

import pandas as pd

from .io import list_matches, load_match_stats
from .scoring import compute_ppi


def season_ppi_frame(weights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate match-level PPI into a minutes-weighted season table.

    ``season_ppi = sum(ppi * minutes) / sum(minutes)`` per player.
    """
    entries = list_matches()
    cols = ["rank", "name", "profile", "season_ppi", "matches", "minutes", "match_ids"]
    if not entries:
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    for entry in entries:
        match_id = entry["id"]
        df = load_match_stats(match_id=match_id)
        scored, _, _ = compute_ppi(df, weights_df)
        for _, r in scored.iterrows():
            rows.append(
                {
                    "name": r["name"],
                    "profile": r["profile"],
                    "match_id": match_id,
                    "ppi": float(r["ppi"]),
                    "playing_time": float(r["playing_time"]),
                }
            )

    long = pd.DataFrame(rows)
    if long.empty:
        return pd.DataFrame(columns=cols)

    long["weighted"] = long["ppi"] * long["playing_time"]
    season = long.groupby("name", as_index=False).agg(
        profile=("profile", "first"),
        minutes=("playing_time", "sum"),
        weighted=("weighted", "sum"),
        matches=("match_id", "nunique"),
        match_ids=("match_id", lambda s: ", ".join(sorted(set(s.astype(str))))),
    )
    season["season_ppi"] = season["weighted"] / season["minutes"]
    season = season.drop(columns=["weighted"])
    season = season.sort_values("season_ppi", ascending=False).reset_index(drop=True)
    season.insert(0, "rank", range(1, len(season) + 1))
    return season[cols]


def season_long_frame(weights_df: pd.DataFrame) -> pd.DataFrame:
    """Per-player per-match PPI for charts."""
    entries = list_matches()
    rows: list[dict] = []
    for entry in entries:
        df = load_match_stats(match_id=entry["id"])
        scored, _, _ = compute_ppi(df, weights_df)
        for _, r in scored.iterrows():
            rows.append(
                {
                    "name": r["name"],
                    "profile": r["profile"],
                    "match_id": entry["id"],
                    "match_label": entry.get("label", entry["id"]),
                    "ppi": float(r["ppi"]),
                    "playing_time": float(r["playing_time"]),
                }
            )
    return pd.DataFrame(rows)
