# Player Performance Index (PPI)

Position-aware soccer player ranking with an interpretable pipeline and a live demo UI.

**Demo case:** Barcelona **4–3** Real Madrid — El Clásico, 11 May 2025 (La Liga MW35), stats mapped from [FBref](https://fbref.com/en/matches/f1804d9c/El-Clasico-Barcelona-Real-Madrid-May-11-2025-La-Liga).

## Techniques (what this project shows)

| Step | Technique | Where |
|------|-----------|--------|
| 1 | Domain importance weights per position | `src/ppi/weights.py` |
| 2 | MinMax feature scaling | `src/ppi/scoring.py` |
| 3 | Position-conditioned score `w · x̃` | `src/ppi/scoring.py` |
| 4 | Cohort rescale to 0–10 | `src/ppi/scoring.py` |
| 5 | LinearRegression surrogate per profile | `src/ppi/models.py` |
| 6 | Feature attribution `wᵢ · xᵢ` | `src/ppi/explain.py` |
| 7 | Squad ranking | `src/ppi/ranking.py` |

```text
x̃ = MinMax(x)
PPI_raw = w_profile · x̃
PPI     = rescale(PPI_raw) × 10
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py
python scripts/score_match.py
streamlit run app/streamlit_app.py
```

Open http://localhost:8501

### Docker

```bash
docker compose up --build
```

## App tabs

1. **Squad ranking** — Barça XI/bench ranked by PPI  
2. **Why this PPI?** — contribution breakdown for a selected player  
3. **Position lens** — same stats scored under every role weight profile  
4. **What-if** — sliders → live linear-surrogate prediction  
5. **Method** — formula, steps, and weight charts  

## Project layout

```text
app/streamlit_app.py     # showcase UI
src/ppi/                 # library (weights, scoring, models, explain)
data/raw/                # El Clásico CSV + importance weights + match meta
data/processed/          # synthetic training corpus + ranked output
scripts/train.py         # fit scaler + per-profile linear models
scripts/score_match.py   # CLI ranking for the demo match
tests/                   # unit tests for the core math
```

## Data notes

- Match CSV columns follow the PPI schema (`playing_time`, `goals`, …, `outcome`).
- Possession share ≈ player touches / team touches (602).
- Turnovers ≈ miscontrols + dispossessions.
- Goalkeeper saves taken from FBref GK shot-stopping (2 saves).
- Training CSV combines the match rows with controlled synthetic perturbations so each profile has enough samples for the surrogate.

## Tests

```bash
PYTHONPATH=src pytest -q
```

## Legacy

The old Anvil uplink scripts (`player_performance_index.py`, `ppi_app.py`) are deprecated stubs. Hardcoded Anvil / DB credentials from the 2023 Colab export were removed — rotate any keys that were ever committed.
