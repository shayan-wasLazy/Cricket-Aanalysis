# SupeRR Selector — Pressure Modelling Lab

A reproducible research workspace for comparing a transparent, hand-weighted cricket Pressure Index (PI) with an XGBoost classifier trained on explicitly synthetic pressure labels.

> **Interpretation guardrail:** XGBoost learns a deliberately constructed synthetic teacher rule. Holdout metrics tell us how well it learns that rule—not whether the rule is the true cricket-pressure definition. Applying it to historical states measures model agreement with the PI feature space, not ground-truth validity.

## Project contents

- `notebooks/SupeRR_Pressure_Modelling_Comparison.ipynb` — executed end-to-end notebook with analysis, synthetic experiment, metrics and plots.
- `cricket_pressure/` — reusable feature engineering, PI, synthetic generation and analysis modules.
- `dashboard/app.py` — Streamlit dashboard for historical PI/XGBoost comparisons, synthetic diagnostics, scenario exploration and PI weight sensitivity.
- `tests/` — lightweight unit tests for pressure logic.
- `data/raw/` — small competition metadata tables are included. The full competition data pack is not vendored in this public repository; see the setup instructions below.
- `data/synthetic/`, `data/processed/`, `outputs/`, `models/` — generated when the pipeline is run. They are included in the downloadable project bundle.

## Required source data

Obtain the authorized SupeRR Selector data pack and place these files in `data/raw/`:

- `ball_by_ball.csv` or `ball_by_ball.csv.gz`
- `matches.csv`
- `competitions.csv`
- `teams.csv`
- `playing_eleven.csv`
- `competition_team_player_mapping.csv`

The three small metadata CSVs (`matches.csv`, `competitions.csv`, and `teams.csv`) are included as convenience copies. The ball-by-ball, playing-XI and player-mapping files are not committed to the public repo. Do not redistribute competition data unless its terms permit it.

## Setup

Python 3.10–3.13 is expected.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the analysis

From the repository root:

```bash
python run_analysis.py
```

This builds delivery-level match states, PI features, synthetic scenarios, the XGBoost classifier, comparisons, CSV outputs, metric JSON files and PNG charts.

Open the notebook:

```bash
jupyter lab notebooks/SupeRR_Pressure_Modelling_Comparison.ipynb
```

Launch the dashboard after running the pipeline:

```bash
streamlit run dashboard/app.py
```

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

Death-over pressure overlaps conceptually with overall time urgency; sensitivity analysis is included to show how much that design choice affects scores.

## Synthetic XGBoost experiment

The generator samples plausible chase states, creates a declared nonlinear teacher score with interactions and small noise, and bins it into Low / Moderate / High / Extreme classes. XGBoost is trained on 78% and evaluated on a stratified 22% holdout. The hand-designed PI baseline is evaluated on the exact same synthetic holdout.

XGBoost's stronger synthetic metrics are expected because the synthetic teacher includes nonlinear interactions that a fixed weighted sum cannot reproduce. This is a controlled algorithm experiment—not evidence that XGBoost is more valid for real cricket.

## Important limitations

1. The supplied `extras1_runs` field appears to be a categorical code, not numeric runs. Bye/legbye run values are unresolved for 3,166 deliveries in the selected innings. Reconcile reconstructed innings totals against official scorecards before trusting the downstream chase states.
2. Tournament points tables are not part of the pack. The project does not claim to identify mathematically must-win group matches; match stakes are a transparent stage/knockout proxy only.
3. Revised overs and targets are applied where available, but shortened innings and score reconstruction still need scorecard validation.
4. Player pressure summaries are descriptive comparisons, not final top-10 rankings.
5. No independent pressure ground-truth labels exist. Historical PI–XGBoost correlation is agreement, not validation.
6. The synthetic holdout classification report measures fidelity to our chosen synthetic rule and should never be presented as real-match predictive performance.

## Next research steps

- Reconcile scores and extras against official scorecards.
- Add tournament standings and qualification rules to derive genuine must-win states.
- Train a separate historical win-probability model on outcomes, grouped by match and evaluated chronologically.
- Assess PI monotonicity, match/season stability, sensitivity and player-ranking stability before submitting final rankings.

## Reproducibility

The default random seed is `42`. Run `python run_analysis.py` to regenerate all analysis outputs. Detailed results are written to `outputs/metrics/experiment_metrics.json`.
