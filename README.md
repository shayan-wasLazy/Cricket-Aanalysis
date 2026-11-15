# SupeRR Selector — Pressure Modelling Lab

A reproducible research workspace to compare a transparent, hand-weighted cricket Pressure Index (PI) with an XGBoost classifier trained on **synthetically generated** pressure labels.

> **Interpretation guardrail:** the XGBoost model is trained on a synthetic teacher rule, not real match outcomes or human pressure labels. Its synthetic holdout metrics measure how well it learns that rule. Applying it to historical matches is an agreement comparison against the PI feature space, not proof that the XGBoost score is a better or more truthful pressure measure.

## What is included

- `notebooks/SupeRR_Pressure_Modelling_Comparison.ipynb` — end-to-end notebook with feature engineering, PI, synthetic generation, XGBoost training, diagnostics, comparisons, scenario tests, plots and exports.
- `cricket_pressure/` — reusable data engineering, PI and synthetic XGBoost code.
- `dashboard/app.py` — Streamlit decision dashboard with historical comparisons, model diagnostics, scenario playground and adjustable PI weights.
- `data/raw/` — supplied competition, match, team, playing-XI, player mapping and compressed ball-by-ball CSV files.
- `data/synthetic/` — reproducible synthetic scenarios and XGBoost holdout predictions.
- `outputs/` — summaries, comparison tables, feature importance and graphs.
- `models/synthetic_pressure_xgb.json` — trained synthetic-label XGBoost classifier.

## Setup

Python 3.10–3.13 is expected.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the complete analysis

From the repository root:

```bash
python run_analysis.py
```

This rebuilds the processed state table, synthetic dataset, model, diagnostic metrics, comparison CSVs and PNG charts under `data/processed/`, `data/synthetic/`, `models/` and `outputs/`.

Then open the notebook:

```bash
jupyter lab notebooks/SupeRR_Pressure_Modelling_Comparison.ipynb
```

Or launch the dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard expects `python run_analysis.py` to have been run first.

## Pressure Index v1 — hand-designed baseline

| Component | Weight | Definition |
|---|---:|---|
| Required run rate | 25.0% | `clip(required_run_rate / 12, 0, 1)` |
| Wickets lost | 20.0% | `1 - wickets_remaining / 10` |
| Innings time urgency | 15.0% | `1 - balls_remaining / scheduled_balls` |
| Momentum shortfall | 15.0% | `clip(max(RRR - recent_RR, 0) / 12, 0, 1)` |
| Death-over pressure | 10.0% | `clip(1 - balls_remaining / 30, 0, 1)` |
| Target difficulty | 7.5% | First-innings target percentile within format/season |
| Match stakes | 7.5% | Stage/knockout heuristic from supplied match metadata |

The components are combined as a weighted average and multiplied by 100. The death-over feature adds a deliberate last-five-overs stress signal; it is related to overall time urgency, so sensitivity tests are included to monitor potential double-counting.

## Synthetic XGBoost experiment

The synthetic generator samples plausible chase states (RRR, wickets, balls, recent run rate, target difficulty, match stakes and format), creates a **declared nonlinear teacher score with interactions and small noise**, then bins it into Low / Moderate / High / Extreme classes. XGBoost is trained on 78% and evaluated on a stratified 22% holdout. Predicted class probabilities are converted into an expected score using class midpoints `[15, 45, 70, 90]`.

The synthetic model is intentionally an experiment in learning a more flexible decision surface. It is not trained on genuine labels and must not be called a learned ground-truth pressure model.

## Important dataset limitations

1. `extras1_runs` is a categorical code in the supplied file, not a numeric run count. The pack does not fully disambiguate bye/legbye run values. The code therefore creates provisional innings totals and flags **3,166 unresolved bye/legbye deliveries** in the selected source data. Reconcile totals against official scorecards before relying on rankings or final results.
2. Tournament standings/points tables are not included. The project does **not** claim to know whether an ordinary group-stage match was mathematically must-win. `match_stakes` is only a transparent stage/knockout proxy; exact must-win status remains unavailable.
3. Some first/second innings may be shortened or affected by revised targets/overs. Revised fields are applied when present, but the source data still needs independent scorecard validation.
4. Player summaries are descriptive pressure exposure views, not final player rankings. Bowler run-concession metrics remain provisional where byes/legbyes are unresolved.
5. Historical PI–XGBoost correlation is model agreement only. It is not a validation metric because there is no independent pressure ground truth in the supplied data.

## Recommended next research steps

- Reconcile reconstructed innings totals against official scorecards.
- Add points tables and qualification rules to derive true must-win scenarios.
- Build a separate historical win-probability model from match outcomes (grouped by match and tested chronologically). Do not reuse synthetic labels for this.
- Validate PI monotonicity, match-level stability, player ranking stability and season/competition transfer before producing final top-10s.

## Reproducibility

The default random seed is `42`. Run `python run_analysis.py` to regenerate the synthetic data and model deterministically (subject to library/platform version differences). `outputs/metrics/experiment_metrics.json` records the latest run metrics.