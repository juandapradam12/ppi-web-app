# Player Performance Index (PPI)

[![CI](https://github.com/juandapradam12/ppi-web-app/actions/workflows/ci.yml/badge.svg)](https://github.com/juandapradam12/ppi-web-app/actions/workflows/ci.yml)

Position-aware soccer player ranking with an interpretable pipeline and a live Streamlit demo.

**Demo matches (FBref, held out from surrogate training):**

| Match | Result |
|-------|--------|
| [La Liga · El Clásico (May 2025)](https://fbref.com/en/matches/f1804d9c/El-Clasico-Barcelona-Real-Madrid-May-11-2025-La-Liga) | Barça **4–3** |
| [Supercopa semi (Jan 2025)](https://fbref.com/en/matches/0c126206/El-Clasico-Real-Madrid-Barcelona-January-12-2025-Supercopa-de-Espana) | Barça **2–5** |

## Techniques

| Step | Technique | Module |
|------|-----------|--------|
| 1 | Domain weights per position | `src/ppi/weights.py` |
| 2 | MinMax scaling | `src/ppi/scoring.py` |
| 3 | Position score `w · x̃` | `src/ppi/scoring.py` |
| 4 | Cohort rescale → 0–10 | `src/ppi/scoring.py` |
| 5 | Linear surrogate (targets **raw** PPI) | `src/ppi/models.py` |
| 6 | Holdout validation vs expert | `src/ppi/validation.py` |
| 7 | Attribution `wᵢ · xᵢ` | `src/ppi/explain.py` |

```text
x̃ = MinMax(x)
PPI_raw = w_profile · x̃
PPI     = rescale(PPI_raw) × 10   # within match squad
```

Real match CSVs are **not** in the synthetic training set; surrogates are evaluated on holdout squads (see **Surrogate check** in the app).

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
5. **Surrogate check** — holdout scatter + MAE / Spearman  
6. **Weight lab** — tweak goals / recoveries / saves weights  
7. **Method** — formula, steps, weight charts  

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
