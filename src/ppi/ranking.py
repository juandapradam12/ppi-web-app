"""Squad ranking helpers."""

from __future__ import annotations

import pandas as pd


def rank_players(
    scored: pd.DataFrame,
    *,
    name_col: str = "name",
    profile_col: str = "profile",
    score_col: str = "ppi",
    min_minutes: float | None = None,
) -> pd.DataFrame:
    """
    Aggregate to one row per player (mean PPI across matches if needed) and rank.

    When ``min_minutes`` is set, rows with playing_time below the threshold are
    dropped before aggregation — useful for match demos.
    """
    frame = scored.copy()
    if min_minutes is not None and "playing_time" in frame.columns:
        frame = frame.loc[frame["playing_time"] >= min_minutes]

    if frame.empty:
        return pd.DataFrame(columns=[name_col, profile_col, score_col, "rank"])

    grouped = (
        frame.groupby(name_col, as_index=False)
        .agg({score_col: "mean", profile_col: "first", **(
            {"playing_time": "sum"} if "playing_time" in frame.columns else {}
        )})
    )
    grouped = grouped.sort_values(score_col, ascending=False).reset_index(drop=True)
    grouped.insert(0, "rank", range(1, len(grouped) + 1))
    cols = ["rank", name_col, profile_col, score_col]
    if "playing_time" in grouped.columns:
        cols.append("playing_time")
    return grouped[cols]
