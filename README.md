# Player Performance Index (PPI)

[![CI](https://github.com/juandapradam12/ppi-web-app/actions/workflows/ci.yml/badge.svg)](https://github.com/juandapradam12/ppi-web-app/actions/workflows/ci.yml)

**Turn raw match stats into clear, role-aware player rankings — explainable enough for the pitch room, rigorous enough for a data science portfolio.**

PPI combines **domain expertise** (position-specific importance weights) with **transparent machine learning** (Ridge surrogates, calibration, pairwise ranking) so every score can be justified: *why this player, why this rank, how sure are we?*

Demo on real Barcelona fixtures (El Clásico, Supercopa, Celta) mapped from FBref — squad boards, contribution breakdowns, what-if analysis, uncertainty bands, and season aggregation in one Streamlit app.

> **Who it’s for:** scouts & performance analysts who need interpretable rankings · clubs exploring decision-support prototypes · DS/ML engineers showcasing end-to-end sports analytics.

**Demo matches (held out from surrogate training):**

| Match | Result |
|-------|--------|
| [La Liga · El Clásico (May 2025)](https://fbref.com/en/matches/f1804d9c/El-Clasico-Barcelona-Real-Madrid-May-11-2025-La-Liga) | Barça **4–3** |
| [Supercopa semi (Jan 2025)](https://fbref.com/en/matches/0c126206/El-Clasico-Real-Madrid-Barcelona-January-12-2025-Supercopa-de-Espana) | Barça **2–5** |
| La Liga · vs Celta (Apr 2025) | Barça **4–3** |

## Techniques

| Step | Technique | Module |
|------|-----------|--------|
| 1 | Domain weights per position | `src/ppi/weights.py` |
| 2 | MinMax scaling | `src/ppi/scoring.py` |
| 3 | Position score `w · x̃` | `src/ppi/scoring.py` |
| 4 | Cohort rescale → 0–10 | `src/ppi/scoring.py` |
| 5 | Surrogate stack: OLS → **RidgeCV** → **isotonic** (+ **LMO** metrics) | `src/ppi/models.py` |
| 6 | Holdout validation (3 variants + leave-one-match-out) | `src/ppi/validation.py` |
| 7 | Minutes-weighted season PPI | `src/ppi/season.py` |
| 8 | Bootstrap / conformal uncertainty | `src/ppi/uncertainty.py` |
| 9 | Model B pairwise ranking | `src/ppi/ranking_model.py` |
| 10 | Attribution `wᵢ · xᵢ` | `src/ppi/explain.py` |

```text
x̃ = MinMax(x)
PPI_raw = w_profile · x̃
PPI     = rescale(PPI_raw) × 10   # within match squad
```

Real match CSVs are **not** in the synthetic training set. **Surrogate check** reports:

1. **In-sample calibrator** fit on all demo matches (optimistic).
2. **Leave-one-match-out** isotonic (honest): calibrate on the other matches, score the left-out squad.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --rebuild-data
streamlit run app/streamlit_app.py
```

```bash
python scripts/score_match.py --match-id el-clasico-2025-05-11
python scripts/ingest_fbref.py --template   # FBref → PPI column guide
```

### Docker

```bash
docker compose up --build
```

## App sections

1. **Squad ranking** — ranked table + CSV export  
2. **Why this PPI?** — contribution breakdown  
3. **Position lens** — same stats, different role weights  
4. **What-if** — expert vs surrogate PPI side by side (default: Raphinha)  
5. **Surrogate check** — OLS / Ridge / calibrated + **leave-one-match-out**  
6. **Uncertainty** — bootstrap PPI bands + top-3 probability  
7. **Model B** — pairwise ranking vs expert ranks  
8. **Season view** — minutes-weighted PPI across demo matches  
9. **Weight lab** — tweak goals / recoveries / saves weights  
10. **Method** — formula, steps, weight charts  

Deep links: `/?page=Surrogate%20check&match=el-clasico-2025-05-11`

## Project layout

```text
app/streamlit_app.py
notebooks/01_ppi_walkthrough.ipynb
src/ppi/
data/raw/matches.json          # match registry
data/processed/training_players.csv  # synthetic only (generated)
scripts/build_training_data.py
scripts/train.py
scripts/ingest_fbref.py
tests/
legacy/                        # old Anvil/Colab entrypoints (archived)
```

## Models

Fitted artifacts live under `models/` locally (gitignored). CI and Docker run `scripts/train.py` on each build.

## Tests

```bash
PYTHONPATH=src pytest -q
```

## Legacy

Original Anvil uplink scripts are archived under `legacy/`. Hardcoded credentials from the 2023 export were removed — rotate any keys that were ever committed.
