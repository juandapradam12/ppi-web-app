"""
PPI Showcase — Streamlit interface for the Player Performance Index.

Demo case: Barcelona 4–3 Real Madrid (El Clásico, 11 May 2025) via FBref.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppi.explain import feature_contributions
from ppi.io import DATA_RAW, MODELS_DIR, list_matches, load_match_meta, load_match_stats
from ppi.models import PositionModelBundle
from ppi.ranking import rank_players
from ppi.schema import FEATURE_COLUMNS, FEATURE_LABELS, PROFILES
from ppi.scoring import compute_ppi, scale_features
from ppi.season import season_long_frame, season_ppi_frame
from ppi.validation import compare_expert_vs_variants, holdout_metrics
from ppi.weights import apply_weight_multipliers, load_weights, weight_vector

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

NAVY = "#071527"
CRIMSON = "#C8102E"
GOLD = "#E8B84A"
PITCH = "#1F6B4A"
FOG = "#E8EEF6"
INK = "#0E1A2B"

CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
  font-family: 'Manrope', sans-serif;
  color: {INK};
}}

.stApp {{
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(200,16,46,0.18), transparent 55%),
    radial-gradient(900px 500px at 95% 5%, rgba(31,107,74,0.22), transparent 50%),
    linear-gradient(165deg, #f4f7fb 0%, #e7eef7 42%, #dfe8df 100%);
  background-attachment: fixed;
}}

.block-container {{
  padding-top: 1.4rem;
  padding-bottom: 3rem;
  max-width: 1180px;
}}

.ppi-hero {{
  position: relative;
  overflow: hidden;
  border-radius: 0;
  margin: 0 0 1.6rem 0;
  padding: 2.4rem 1.6rem 2.1rem;
  background:
    linear-gradient(115deg, rgba(7,21,39,0.94) 0%, rgba(7,21,39,0.78) 48%, rgba(200,16,46,0.55) 100%),
    radial-gradient(circle at 80% 30%, rgba(232,184,74,0.25), transparent 40%);
  color: white;
  animation: rise 0.8s ease-out both;
}}

.ppi-hero::after {{
  content: "";
  position: absolute;
  inset: auto -10% -40% 40%;
  height: 180%;
  background:
    repeating-linear-gradient(
      -18deg,
      transparent,
      transparent 14px,
      rgba(255,255,255,0.04) 14px,
      rgba(255,255,255,0.04) 15px
    );
  pointer-events: none;
}}

.ppi-brand {{
  font-family: 'Bebas Neue', sans-serif;
  font-size: clamp(3.4rem, 9vw, 5.8rem);
  letter-spacing: 0.04em;
  line-height: 0.92;
  margin: 0;
  animation: brand-glow 2.8s ease-in-out infinite alternate;
}}

.ppi-kicker {{
  font-size: 0.78rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: {GOLD};
  margin: 0 0 0.55rem 0;
  font-weight: 700;
}}

.ppi-sub {{
  max-width: 36rem;
  margin: 0.85rem 0 0;
  font-size: 1.02rem;
  line-height: 1.45;
  color: rgba(255,255,255,0.88);
}}

.ppi-match {{
  display: inline-block;
  margin-top: 1.15rem;
  font-size: 0.86rem;
  letter-spacing: 0.04em;
  color: rgba(255,255,255,0.75);
  border-left: 3px solid {GOLD};
  padding-left: 0.75rem;
}}

.section-title {{
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2.1rem;
  letter-spacing: 0.03em;
  margin: 0.2rem 0 0.35rem;
  color: {NAVY};
}}

.section-lead {{
  color: #445266;
  margin-bottom: 1.1rem;
  max-width: 42rem;
}}

.metric-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.85rem;
  margin: 0.4rem 0 1.4rem;
  animation: rise 0.9s 0.12s ease-out both;
}}

.metric-cell {{
  padding: 0.85rem 0.9rem;
  background: rgba(255,255,255,0.55);
  border-bottom: 3px solid {CRIMSON};
}}

.metric-cell .label {{
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #5a6a7e;
  font-weight: 700;
}}

.metric-cell .value {{
  font-family: 'Bebas Neue', sans-serif;
  font-size: 1.85rem;
  color: {NAVY};
  line-height: 1.1;
  margin-top: 0.2rem;
}}

.formula-box {{
  background: {NAVY};
  color: white;
  padding: 1.2rem 1.35rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.95rem;
  line-height: 1.55;
  margin: 0.6rem 0 1.2rem;
  animation: rise 0.7s ease-out both;
}}

.step-row {{
  display: grid;
  grid-template-columns: 3.2rem 1fr;
  gap: 0.85rem;
  margin-bottom: 1rem;
  align-items: start;
}}

.step-num {{
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2rem;
  color: {CRIMSON};
  line-height: 1;
}}

div[data-testid="stTabs"] button {{
  font-family: 'Manrope', sans-serif;
  font-weight: 700;
}}

@keyframes rise {{
  from {{ opacity: 0; transform: translateY(14px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes brand-glow {{
  from {{ text-shadow: 0 0 0 rgba(232,184,74,0); }}
  to {{ text-shadow: 0 0 28px rgba(232,184,74,0.35); }}
}}

@media (max-width: 800px) {{
  .metric-strip {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .ppi-hero {{ padding: 1.8rem 1.1rem 1.6rem; }}
}}
"""


@st.cache_data
def _load_match_bundle(match_id: str):
    match = load_match_stats(match_id=match_id)
    meta = load_match_meta(match_id=match_id)
    weights = load_weights(DATA_RAW / "importance_weights.csv")
    scored, scaler, _ = compute_ppi(match, weights)
    return match, meta, weights, scored, scaler


def _expert_ppi_for_inputs(
    inputs: dict[str, float],
    profile: str,
    weights_df: pd.DataFrame,
    cohort: pd.DataFrame,
) -> float:
    """Expert PPI for a what-if row, rescaled within cohort + hypothetical player."""
    base_cols = [c for c in cohort.columns if c not in ("ppi", "ppi_raw")]
    row = {col: float(inputs[col]) for col in FEATURE_COLUMNS}
    row.update({"name": "__whatif__", "profile": profile})
    if "match_id" in cohort.columns:
        row["match_id"] = cohort["match_id"].iloc[0]
    trial = pd.concat([cohort[base_cols], pd.DataFrame([row])], ignore_index=True)
    rescored, _, _ = compute_ppi(trial, weights_df)
    return float(rescored.loc[rescored["name"] == "__whatif__", "ppi"].iloc[0])


@st.cache_resource
def _load_models() -> PositionModelBundle | None:
    if not (MODELS_DIR / "metrics.json").exists():
        return None
    return PositionModelBundle.load(MODELS_DIR)


def _plotly_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.35)",
        font=dict(family="Manrope, sans-serif", color=INK),
        margin=dict(l=20, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(7,21,39,0.08)")
    fig.update_yaxes(gridcolor="rgba(7,21,39,0.08)")
    return fig


def render_hero(meta: dict) -> None:
    st.markdown(
        f"""
        <div class="ppi-hero">
          <p class="ppi-kicker">Player Performance Index</p>
          <h1 class="ppi-brand">PPI</h1>
          <p class="ppi-sub">
            Position-aware player ranking: expert weights define the score,
            a transparent linear model approximates it live — with clear feature attribution.
          </p>
          <div class="ppi-match">
            Demo case · {meta['home_team']} {meta['score']} {meta['away_team']}<br/>
            {meta['competition']} · {meta['date']} · source: FBref
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tab_squad(scored: pd.DataFrame, weights: pd.DataFrame, meta: dict) -> None:
    st.markdown('<h2 class="section-title">Squad ranking board</h2>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-lead">Barcelona lineup from El Clásico ({meta["score"]}), '
        "scored with role-specific importance weights then ranked by PPI (0–10).</p>",
        unsafe_allow_html=True,
    )

    min_minutes = st.slider("Minimum minutes played", 0, 90, 30, 5)
    profile_filter = st.multiselect("Profiles", list(PROFILES), default=list(PROFILES))

    ranked = rank_players(scored, min_minutes=float(min_minutes))
    ranked = ranked[ranked["profile"].isin(profile_filter)]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Players ranked", len(ranked))
    with c2:
        st.metric("Top PPI", f"{ranked['ppi'].iloc[0]:.2f}" if len(ranked) else "—")
    with c3:
        top_name = ranked["name"].iloc[0] if len(ranked) else "—"
        st.metric("Match MVP (PPI)", top_name)
    with c4:
        st.metric("Result", meta["score"] + " W")

    if ranked.empty:
        st.info("No players match the current filters.")
        return

    fig = px.bar(
        ranked.sort_values("ppi"),
        x="ppi",
        y="name",
        color="profile",
        orientation="h",
        color_discrete_map={
            "Attacker": CRIMSON,
            "Midfielder": GOLD,
            "Defender": PITCH,
            "Goalkeeper": NAVY,
        },
        labels={"ppi": "PPI (0–10)", "name": "", "profile": "Profile"},
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(_plotly_layout(fig, height=28 * len(ranked) + 120), use_container_width=True)

    show = ranked.copy()
    show["ppi"] = show["ppi"].round(2)
    show = show.rename(columns={"rank": "Rank", "name": "Player", "profile": "Profile", "ppi": "PPI", "playing_time": "Minutes"})
    st.dataframe(show, use_container_width=True, hide_index=True)

    csv_bytes = show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download ranking (CSV)",
        data=csv_bytes,
        file_name=f"ppi_ranking_{meta.get('match_id', 'match')}.csv",
        mime="text/csv",
    )


def tab_player(scored: pd.DataFrame, weights: pd.DataFrame, scaler: MinMaxScaler) -> None:
    st.markdown('<h2 class="section-title">Why this PPI?</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-lead">Click into a player to see signed feature contributions '
        "<code>wᵢ · xᵢ</code> — the building blocks of the raw score before cohort rescale.</p>",
        unsafe_allow_html=True,
    )

    eligible = scored.sort_values("ppi", ascending=False)
    player = st.selectbox("Player", eligible["name"].tolist())
    row = eligible.loc[eligible["name"] == player].iloc[0]

    scaled, _ = scale_features(scored, scaler=scaler, fit=False)
    scaled_row = scaled.loc[row.name]

    contrib = feature_contributions(scaled_row, row["profile"], weights)
    top = contrib.head(8)

    m1, m2, m3 = st.columns(3)
    m1.metric("PPI", f"{row['ppi']:.2f}")
    m2.metric("Profile", row["profile"])
    m3.metric("Minutes", f"{row['playing_time']:.0f}")

    fig = go.Figure(
        go.Bar(
            x=top["contribution"],
            y=top["label"],
            orientation="h",
            marker_color=[CRIMSON if v >= 0 else "#5B6B7C" for v in top["contribution"]],
            hovertemplate="%{y}<br>contribution=%{x:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Largest feature contributions")
    st.plotly_chart(_plotly_layout(fig, height=420), use_container_width=True)

    with st.expander("Full contribution table"):
        pretty = contrib.copy()
        pretty["weight"] = pretty["weight"].round(3)
        pretty["scaled_value"] = pretty["scaled_value"].round(3)
        pretty["contribution"] = pretty["contribution"].round(4)
        st.dataframe(
            pretty[["label", "weight", "scaled_value", "contribution"]].rename(
                columns={
                    "label": "Feature",
                    "weight": "Weight",
                    "scaled_value": "Scaled value",
                    "contribution": "Contribution",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def tab_lens(scored: pd.DataFrame, weights: pd.DataFrame) -> None:
    st.markdown('<h2 class="section-title">Position lens</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-lead">Same raw stats, different role weights. '
        "Shows how profile priors change the score — core to the method.</p>",
        unsafe_allow_html=True,
    )

    player = st.selectbox("Player to re-score", scored.sort_values("ppi", ascending=False)["name"].tolist(), key="lens_player")
    base = scored.loc[scored["name"] == player].iloc[0]

    rows = []
    for profile in PROFILES:
        trial = scored.copy()
        # replace only this player's profile for a fair single-player probe on the cohort scaler
        trial.loc[trial["name"] == player, "profile"] = profile
        rescored, _, _ = compute_ppi(trial, weights)
        ppi = float(rescored.loc[rescored["name"] == player, "ppi"].iloc[0])
        rows.append({"profile": profile, "ppi": ppi, "is_actual": profile == base["profile"]})

    lens = pd.DataFrame(rows)
    fig = px.bar(
        lens,
        x="profile",
        y="ppi",
        color="is_actual",
        color_discrete_map={True: CRIMSON, False: NAVY},
        labels={"profile": "Scored as", "ppi": "PPI", "is_actual": "Actual profile"},
    )
    st.plotly_chart(_plotly_layout(fig, height=380), use_container_width=True)
    st.caption(f"Actual profile for {player}: **{base['profile']}** (PPI {base['ppi']:.2f}).")


def tab_whatif(bundle: PositionModelBundle | None, weights: pd.DataFrame, scored: pd.DataFrame) -> None:
    st.markdown('<h2 class="section-title">What-if calculator</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-lead">Adjust match stats and get a live PPI estimate from the '
        "position-specific linear surrogate (trained to approximate the expert index).</p>",
        unsafe_allow_html=True,
    )

    if bundle is None:
        st.warning("Models not found. Run `python scripts/train.py` first.")
        return

    player_options = scored.sort_values("ppi", ascending=False)["name"].tolist()
    default_idx = player_options.index("Raphinha") if "Raphinha" in player_options else 0
    seed_name = st.selectbox(
        "Start from player",
        ["Custom blank"] + player_options,
        index=default_idx + 1 if player_options else 0,
    )
    if seed_name == "Custom blank":
        seed = {c: 0.0 for c in FEATURE_COLUMNS}
        seed.update(
            {
                "playing_time": 90,
                "success_rate": 85,
                "possession_participation": 8,
                "touches": 50,
                "completed_passes": 40,
                "outcome": 1,
            }
        )
        default_profile = "Attacker"
    else:
        seed_row = scored.loc[scored["name"] == seed_name].iloc[0]
        seed = {c: float(seed_row[c]) for c in FEATURE_COLUMNS}
        default_profile = seed_row["profile"]

    profile = st.selectbox("Profile", list(PROFILES), index=list(PROFILES).index(default_profile))

    left, right = st.columns(2)
    inputs: dict[str, float] = {}
    with left:
        inputs["playing_time"] = st.slider("Playing time (min)", 1, 90, int(seed["playing_time"]))
        inputs["goals"] = st.number_input("Goals", 0, 5, int(seed["goals"]))
        inputs["assists"] = st.number_input("Assists", 0, 5, int(seed["assists"]))
        inputs["key_passes"] = st.number_input("Key passes", 0, 15, int(seed["key_passes"]))
        inputs["completed_passes"] = st.number_input("Completed passes", 0, 120, int(seed["completed_passes"]))
        inputs["success_rate"] = st.slider("Pass completion %", 0.0, 100.0, float(seed["success_rate"]))
        inputs["possession_participation"] = st.slider("Possession share %", 0.0, 25.0, float(seed["possession_participation"]))
        inputs["touches"] = st.number_input("Touches", 0, 150, int(seed["touches"]))
        inputs["shots"] = st.number_input("Shots", 0, 15, int(seed["shots"]))
        inputs["shots_on_goal"] = st.number_input("Shots on target", 0, 10, int(seed["shots_on_goal"]))
    with right:
        inputs["crosses"] = st.number_input("Crosses", 0, 15, int(seed["crosses"]))
        inputs["turnovers"] = st.number_input("Turnovers", 0, 15, int(seed["turnovers"]))
        inputs["recoveries"] = st.number_input("Recoveries", 0, 20, int(seed["recoveries"]))
        inputs["fouls"] = st.number_input("Fouls", 0, 8, int(seed["fouls"]))
        inputs["yellow_cards"] = st.number_input("Yellow cards", 0, 2, int(seed["yellow_cards"]))
        inputs["red_cards"] = st.number_input("Red cards", 0, 1, int(seed["red_cards"]))
        inputs["saves"] = st.number_input("Saves", 0, 12, int(seed["saves"]))
        inputs["clean_sheets"] = st.selectbox("Clean sheet", [0, 1], index=int(seed["clean_sheets"]))
        inputs["offsides"] = st.number_input("Offsides", 0, 5, int(seed["offsides"]))
        inputs["outcome"] = st.selectbox("Team won?", [1, 0], index=0 if seed["outcome"] >= 0.5 else 1, format_func=lambda x: "Yes" if x == 1 else "No")

    expert_live = _expert_ppi_for_inputs(inputs, profile, weights, scored)
    pred_ols = bundle.predict_display(profile, inputs, variant="ols")
    pred_ridge = bundle.predict_display(profile, inputs, variant="ridge")
    pred_cal = bundle.predict_row(profile, inputs)
    st.markdown(
        f"""
        <div class="metric-strip">
          <div class="metric-cell"><div class="label">Expert PPI</div><div class="value">{expert_live:.2f}</div></div>
          <div class="metric-cell"><div class="label">Calibrated</div><div class="value">{pred_cal:.2f}</div></div>
          <div class="metric-cell"><div class="label">Ridge</div><div class="value">{pred_ridge:.2f}</div></div>
          <div class="metric-cell"><div class="label">OLS</div><div class="value">{pred_ols:.2f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Δ calibrated − expert: **{pred_cal - expert_live:+.2f}** · G+A: {inputs['goals'] + inputs['assists']:.0f}")

    # Contribution under expert weights using the bundle scaler
    scaled_vec = bundle.scaler.transform(np.array([[inputs[c] for c in FEATURE_COLUMNS]], dtype=float))[0]
    scaled_series = pd.Series(dict(zip(FEATURE_COLUMNS, scaled_vec)))
    contrib = feature_contributions(scaled_series, profile, weights).head(8)
    fig = go.Figure(
        go.Bar(
            x=contrib["contribution"],
            y=contrib["label"],
            orientation="h",
            marker_color=[CRIMSON if v >= 0 else "#5B6B7C" for v in contrib["contribution"]],
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Expert-weight contributions for this what-if")
    st.plotly_chart(_plotly_layout(fig, height=380), use_container_width=True)

    if bundle.metrics:
        with st.expander("Training metrics (synthetic hold-in)"):
            rows = [
                {"profile": k, **v}
                for k, v in bundle.metrics.items()
                if not str(k).startswith("_") and k != "holdout"
            ]
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def tab_surrogate(
    match_df: pd.DataFrame,
    weights: pd.DataFrame,
    bundle: PositionModelBundle | None,
    meta: dict,
) -> None:
    st.markdown('<h2 class="section-title">Surrogate check</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-lead">Holdout squads: compare expert PPI to <strong>OLS</strong>, '
        "<strong>RidgeCV</strong>, and <strong>isotonic-calibrated</strong> Ridge (fit on holdout).</p>",
        unsafe_allow_html=True,
    )
    if bundle is None:
        st.warning("Models not found. Run `python scripts/train.py`.")
        return

    pooled = bundle.metrics.get("holdout_variants", {})
    if pooled:
        st.markdown("#### Pooled holdout metrics (all demo matches)")
        st.dataframe(
            pd.DataFrame(pooled).T.reset_index().rename(columns={"index": "variant"}),
            use_container_width=True,
            hide_index=True,
        )

    comparison = compare_expert_vs_variants(match_df, weights, bundle)
    variant = st.selectbox(
        "Scatter variant",
        ["calibrated", "ridge", "ols"],
        format_func=lambda v: {"calibrated": "Calibrated Ridge", "ridge": "RidgeCV", "ols": "OLS"}[v],
    )
    y_col = {"calibrated": "calibrated_ppi", "ridge": "ridge_ppi", "ols": "ols_ppi"}[variant]
    hm = holdout_metrics(match_df, weights, bundle, variant=variant)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{hm['mae']:.2f}")
    c2.metric("R²", f"{hm['r2']:.3f}")
    c3.metric("Spearman", f"{hm['spearman']:.3f}")
    c4.metric("Players", f"{int(hm['n'])}")

    fig = px.scatter(
        comparison,
        x="expert_ppi",
        y=y_col,
        text="name",
        color="profile",
        color_discrete_map={
            "Attacker": CRIMSON,
            "Midfielder": GOLD,
            "Defender": PITCH,
            "Goalkeeper": NAVY,
        },
        labels={"expert_ppi": "Expert PPI", y_col: variant},
    )
    fig.update_traces(textposition="top center")
    lim = [0, 10.5]
    fig.add_shape(type="line", x0=0, y0=0, x1=10, y1=10, line=dict(dash="dash", color="#5B6B7C"))
    fig.update_xaxes(range=lim)
    fig.update_yaxes(range=lim)
    st.plotly_chart(_plotly_layout(fig, height=440), use_container_width=True)
    st.dataframe(comparison, use_container_width=True, hide_index=True)


def tab_season(weights: pd.DataFrame) -> None:
    st.markdown('<h2 class="section-title">Season view</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-lead">Minutes-weighted PPI across all demo matches in '
        "<code>matches.json</code>: "
        "<code>season_ppi = Σ(ppi×min) / Σ(min)</code>.</p>",
        unsafe_allow_html=True,
    )
    season = season_ppi_frame(weights)
    if season.empty:
        st.info("No matches registered.")
        return

    multi = season[season["matches"] > 1]
    st.metric("Players in pool", len(season))
    if not multi.empty:
        st.markdown("#### Multi-match players")
        st.dataframe(multi, use_container_width=True, hide_index=True)

    st.markdown("#### Season ranking")
    st.dataframe(season, use_container_width=True, hide_index=True)

    long = season_long_frame(weights)
    if long["name"].nunique() >= 1 and long["match_id"].nunique() >= 2:
        pivot_names = multi["name"].tolist() if not multi.empty else season.head(6)["name"].tolist()
        subset = long[long["name"].isin(pivot_names)]
        if not subset.empty:
            fig = px.bar(
                subset,
                x="match_label",
                y="ppi",
                color="name",
                barmode="group",
                labels={"ppi": "Match PPI", "match_label": "Match"},
                title="Match PPI vs season aggregate (selected players)",
            )
            st.plotly_chart(_plotly_layout(fig, height=380), use_container_width=True)


def tab_weight_lab(scored: pd.DataFrame, weights: pd.DataFrame, meta: dict) -> None:
    st.markdown('<h2 class="section-title">Weight lab</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-lead">Nudge a few domain weights and watch the squad ranking move — '
        "shows how expert priors drive the index.</p>",
        unsafe_allow_html=True,
    )

    profile = st.selectbox("Profile weights to edit", list(PROFILES), key="wl_profile")
    g_mult = st.slider("Goals weight ×", 0.5, 2.0, 1.0, 0.05)
    r_mult = st.slider("Recoveries weight ×", 0.5, 2.0, 1.0, 0.05)
    s_mult = st.slider("Saves weight × (GK)", 0.5, 2.0, 1.0, 0.05)

    multipliers = {"goals": g_mult, "recoveries": r_mult, "saves": s_mult}
    tweaked = apply_weight_multipliers(weights, profile, multipliers)

    baseline = rank_players(compute_ppi(scored.copy(), weights)[0], min_minutes=30)
    adjusted = rank_players(compute_ppi(scored.copy(), tweaked)[0], min_minutes=30)

    merged = baseline.merge(
        adjusted[["name", "ppi"]],
        on="name",
        suffixes=("_base", "_adj"),
    )
    merged["delta"] = merged["ppi_adj"] - merged["ppi_base"]

    fig = px.bar(
        merged.sort_values("ppi_adj"),
        x="ppi_adj",
        y="name",
        orientation="h",
        color="profile",
        hover_data=["ppi_base", "delta"],
        labels={"ppi_adj": "PPI (adjusted weights)", "name": ""},
    )
    st.plotly_chart(_plotly_layout(fig, height=28 * len(merged) + 100), use_container_width=True)
    st.caption(
        f"Adjustments apply to **{profile}** weights only (goals ×{g_mult:.2f}, "
        f"recoveries ×{r_mult:.2f}, saves ×{s_mult:.2f})."
    )


def tab_method(weights: pd.DataFrame, meta: dict, bundle: PositionModelBundle | None) -> None:
    st.markdown('<h2 class="section-title">Techniques in plain sight</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-lead">The index is intentionally interpretable. '
        "Each step below is visible in the code under <code>src/ppi/</code>.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="formula-box">
          x̃ = MinMax(x)<br/>
          PPI<sub>raw</sub> = w<sub>profile</sub> · x̃<br/>
          PPI = rescale(PPI<sub>raw</sub>) × 10
        </div>
        """,
        unsafe_allow_html=True,
    )

    steps = [
        ("01", "Domain weights", "Experts (or analysts) set importance per metric and position — attackers reward goals/shots; defenders recoveries/clean sheets; keepers saves."),
        ("02", "Feature scaling", "MinMaxScaler puts heterogeneous stats on [0, 1] so minutes and goals are comparable inside the dot product."),
        ("03", "Position-conditioned score", "The weight vector switches with profile. Same stats ≠ same value as an attacker vs midfielder."),
        ("04", "Cohort display rescale", "Raw scores are stretched to 0–10 within the evaluated group for readable rankings."),
        ("05", "Linear surrogate", "Per-profile LinearRegression learns scaled features → PPI for fast what-if prediction without refitting the full cohort."),
        ("06", "Attribution", "Contributions wᵢ·xᵢ explain why a player ranked where they did."),
    ]
    for num, title, body in steps:
        st.markdown(
            f"""
            <div class="step-row">
              <div class="step-num">{num}</div>
              <div><strong>{title}</strong><br/><span style="color:#445266">{body}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Role weight profiles")
    profile = st.selectbox("Inspect weights for", list(PROFILES), key="method_profile")
    w = weight_vector(weights, profile)
    wdf = pd.DataFrame(
        {
            "feature": [FEATURE_LABELS[c] for c in FEATURE_COLUMNS],
            "weight": w,
        }
    ).sort_values("weight")
    fig = px.bar(
        wdf,
        x="weight",
        y="feature",
        orientation="h",
        color="weight",
        color_continuous_scale=["#5B6B7C", FOG, CRIMSON],
    )
    fig.update_layout(coloraxis_showscale=False, title=f"{profile} importance weights")
    st.plotly_chart(_plotly_layout(fig, height=520), use_container_width=True)

    st.markdown("#### Data source")
    st.write(
        f"Match stats were mapped from the [FBref match report]({meta['source_url']}) "
        f"into the PPI schema. {meta['notes']}"
    )

    if bundle and bundle.metrics.get("holdout_variants"):
        st.markdown("#### Surrogate variants (pooled holdout)")
        st.dataframe(
            pd.DataFrame(bundle.metrics["holdout_variants"]).T.reset_index().rename(columns={"index": "variant"}),
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="PPI · Player Performance Index",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

    matches = list_matches()
    match_labels = {m["id"]: m["label"] for m in matches}

    sections = [
        "Squad ranking",
        "Why this PPI?",
        "Position lens",
        "What-if",
        "Surrogate check",
        "Season view",
        "Weight lab",
        "Method",
    ]
    requested = st.query_params.get("page", sections[0])
    if requested not in sections:
        requested = sections[0]

    with st.sidebar:
        st.markdown("### Match")
        if matches:
            qp_match = st.query_params.get("match")
            ids = [m["id"] for m in matches]
            default_ix = ids.index(qp_match) if qp_match in ids else 0
            selected_id = st.selectbox(
                "Dataset",
                ids,
                index=default_ix,
                format_func=lambda mid: match_labels[mid],
                label_visibility="collapsed",
            )
        else:
            selected_id = None
        st.markdown("### Navigate")
        page = st.radio(
            "Section",
            sections,
            index=sections.index(requested),
            label_visibility="collapsed",
            key="nav_radio",
        )

    match, meta, weights, scored, scaler = _load_match_bundle(selected_id or matches[0]["id"])
    bundle = _load_models()

    if page != requested:
        st.query_params["page"] = page
    if selected_id and st.query_params.get("match") != selected_id:
        st.query_params["match"] = selected_id

    with st.sidebar:
        st.markdown("---")
        st.caption(f"{meta['home_team']} {meta['score']} {meta['away_team']}")
        st.caption(meta["date"])

    render_hero(meta)

    if page == "Squad ranking":
        tab_squad(scored, weights, meta)
    elif page == "Why this PPI?":
        tab_player(scored, weights, scaler)
    elif page == "Position lens":
        tab_lens(scored, weights)
    elif page == "What-if":
        tab_whatif(bundle, weights, scored)
    elif page == "Surrogate check":
        tab_surrogate(match, weights, bundle, meta)
    elif page == "Season view":
        tab_season(weights)
    elif page == "Weight lab":
        tab_weight_lab(scored, weights, meta)
    else:
        tab_method(weights, meta, bundle)

    st.caption(
        "PPI showcase · techniques: role weights · MinMax scaling · linear surrogate · attribution · "
        f"demo data via FBref ({meta['date']})"
    )


if __name__ == "__main__":
    main()
