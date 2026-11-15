# SupeRR Selector — Pressure Modelling Lab

A reproducible research workspace for comparing a transparent, hand-weighted cricket Pressure Index (PI) with an XGBoost classifier trained on explicitly synthetic pressure labels.

> **Interpretation guardrail:** XGBoost learns a deliberately constructed synthetic teacher rule. Holdout metrics tell us how well it learns that rule—not whether the rule is the true cricket-pressure definition. Applying it to historical states measures model agreement with the PI feature space, not ground-truth validity.

## Repository contents

- `notebooks/SupeRR_Pressure_Modelling_Comparison.ipynb` — end-to-end comparison notebook.
- `cricket_pressure/` — feature engineering, PI, synthetic generation and evaluation modules.
- `dashboard/app.py` — Streamlit dashboard for historical comparisons, synthetic diagnostics, scenario exploration and PI weight sensitivity.
- `tests/` — unit tests for the pressure logic.
- `dataset/README.md` — instructions for adding the competition CSVs locally.
- `outputs/`, `models/`, and generated data — created locally by running the analysis; ignored by Git.

## Dataset setup (required before running)

After cloning, download the authorized SupeRR Selector data pack and place these six source files directly inside the tracked `dataset/` folder:

```text
dataset/
├── ball_by_ball.csv
├── matches.csv
├── competitions.csv
├── teams.csv
├── playing_eleven.csv
└── competition_team_player_mapping.csv
```

The loader also accepts gzip-compressed equivalents such as `ball_by_ball.csv.gz` and `matches.csv.gz`.

**The dataset folder is intentionally gitignored.** Its README remains tracked, but CSVs will not be committed or pushed. Keep the source data local unless the competition terms explicitly allow redistribution. See [dataset/README.md](dataset/README.md) for exact steps and an ignore check.

## Clone and install

```bash
git clone https://github.com/shayan-wasLazy/Cricket-Aanalysis.git
cd Cricket-Aanalysis
```

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# Windows cmd
.venv\\Scripts\\activate.bat
# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then add the six CSV files to `dataset/`.

## Run the analysis

From the repository root:

```bash
python run_analysis.py
```

This rebuilds delivery-level match states, PI features, synthetic scenarios, the XGBoost classifier, model comparisons, metrics, CSV exports and PNG charts.

Open the notebook:

```bash
jupyter lab notebooks/SupeRR_Pressure_Modelling_Comparison.ipynb
```

Launch the dashboard after the analysis has generated its outputs:

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

Death-over pressure overlaps conceptually with overall time urgency; sensitivity analysis is included to measure the impact of this design choice.

## Synthetic XGBoost experiment

The generator samples plausible chase states, creates a declared nonlinear teacher score with interactions and small noise, and bins it into Low / Moderate / High / Extreme classes. XGBoost is trained on a stratified train/test split. The hand-designed PI baseline is evaluated on the same synthetic holdout.

XGBoost's stronger synthetic metrics are expected because the synthetic teacher includes nonlinear interactions that a fixed weighted sum cannot reproduce. This is a controlled algorithm experiment—not evidence that XGBoost is more valid for real cricket.

## Important limitations

1. The supplied `extras1_runs` field appears to be a categorical code, not numeric runs. Bye/legbye run values are unresolved for some deliveries. Reconcile reconstructed innings totals against official scorecards before trusting downstream chase states.
2. Tournament points tables are not part of the supplied pack. The project does not claim to identify mathematically must-win group matches; match stakes are a transparent stage/knockout proxy only.
3. Revised overs and targets are applied where available, but shortened innings and score reconstruction still need scorecard validation.
4. Player pressure summaries are descriptive comparisons, not final top-10 rankings.
5. No independent pressure ground-truth labels exist. Historical PI–XGBoost correlation is agreement, not validation.
6. The synthetic holdout classification report measures fidelity to our chosen synthetic rule and must not be presented as real-match predictive performance.

## Reproducibility and Git workflow

- The default random seed is `42`.
- Run `python run_analysis.py` to regenerate results locally.
- Generated outputs, trained model files and `dataset/` contents are ignored by Git.
- Push code, notebook and documentation changes normally; adding local CSVs will not send those files to GitHub.

To verify the dataset ignore rule:

```bash
git check-ignore dataset/ball_by_ball.csv
git status --short
```

The first command should print the data path. The CSVs should not appear as untracked files.

## Research next steps

- Reconcile reconstructed scores and extras against official scorecards.
- Add tournament standings and qualification rules to derive genuine must-win states.
- Train a separate historical win-probability model on actual outcomes, grouped by match and evaluated chronologically.
- Assess PI monotonicity, match/season stability, weight sensitivity and player-ranking stability before producing final rankings.


## New validation workflows

The current pipeline adds four analysis tracks beyond the original PI / synthetic-XGBoost comparison:

### 1. Player sample gates, shrinkage and Top-10 comparison

The default high-pressure definition is a score of at least 60. For each of the following scoring methods, the pipeline builds separate batter and bowler rankings:

- PI baseline
- RRR-heavy PI
- Momentum-heavy PI
- Stakes-heavy PI
- Synthetic-trained XGBoost

Initial eligibility is **at least 100 high-pressure deliveries across at least 10 matches**. Batters are ranked by runs per 100 pressure balls; bowlers by credited wickets per 100 pressure balls. Rates are shrunk toward the role-wide rate with a 100-delivery prior. These are high-pressure efficiency rankings, not yet pressure uplift versus each player's own normal baseline.

Generated files:
- `outputs/player_rankings.csv`
- `outputs/player_top10_comparison.csv`

The dashboard allows the minimum delivery and match thresholds to be changed interactively.

### 2. Match-by-match disagreement review

The pipeline exports:
- `outputs/match_disagreement_summary.csv` — matches ranked by mean and maximum absolute PI/XGBoost score gap.
- `outputs/delivery_disagreements.csv` — up to 500 delivery states with the largest score gaps.

Use the dashboard's Match disagreement tab to inspect pressure trajectories within a selected match.

### 3. Score reconstruction / innings audit

Generated files:
- `outputs/innings_validation.csv` — innings-level provisional totals, legal balls, unresolved extras, wicket counts and schedule checks.
- `outputs/score_reconciliation_audit.csv` — match-level audit. Run-margin comparisons are attempted only for non-DLS run victories with no unresolved bye/legbye deliveries.

The data pack does not include independent official innings totals, so this is a conservative internal/result-text audit, **not a complete official scorecard reconciliation**. Any unresolved extras keep affected matches provisional.

### 4. Historical win-probability model

A second XGBoost model is trained on the actual second-innings outcome: whether the batting team eventually won. It uses pre-delivery match-state features, excludes matches without a valid winner/team ID, and keeps every delivery from one match in the same chronological train or test partition. Match-balanced sample weights reduce the influence of matches with more deliveries.

Metrics include match-weighted ROC-AUC, Brier score, baseline Brier score and log loss, plus a calibration table.

Generated files:
- `models/historical_win_probability_xgb.json`
- `outputs/metrics/historical_win_probability_metrics.json`
- `outputs/historical_win_probability_calibration.csv`
- `outputs/historical_win_probability_feature_importance.csv`
- `data/processed/historical_win_probability_predictions.csv.gz`

This outcome model is an independent empirical benchmark for game-state difficulty. It is **not** a psychological-pressure label and must not be directly interpreted as pressure.

The synthetic confusion matrix is now saved in `outputs/metrics/experiment_metrics.json` and as `outputs/metrics/synthetic_xgb_confusion_matrix.csv`, fixing the previous dashboard display bug.


## Pull future code updates without replacing your local dataset

From your existing clone, first confirm the source CSV is ignored, then fast-forward to the latest code:

```bash
git status --short
git check-ignore dataset/ball_by_ball.csv
git pull --ff-only origin main
```

The `dataset/README.md` file is tracked, but all other files inside `dataset/` are ignored. Therefore your local source CSVs stay on your machine during a normal pull. Generated files under `outputs/`, `models/`, `data/processed/` and `data/synthetic/` are also ignored.

After pulling new analysis code, regenerate outputs and relaunch the dashboard:

```bash
python run_analysis.py
streamlit run dashboard/app.py
```

Do **not** run `git clean -fdx` in this project: that command also deletes ignored files, including your local `dataset/` CSVs and generated analysis artifacts.


## Recency-weighted performance experiments

Recent performances are often more relevant to a current player-selection decision than older performances. The project therefore compares **no decay, 365-day, 540-day and 730-day exponential half-lives** instead of hard-coding a single recency assumption.

For a match age of (a) days and half-life (H):

```text
recency_weight = 2 ** (-age_days / half_life_days)
```

At the half-life, a match receives half the weight of a match on the reference date. At twice the half-life, it receives one-quarter weight.

### Player rankings

For each pressure definition (PI baseline, three PI weight alternatives and synthetic-trained XGBoost), batters and bowlers receive an unweighted ranking plus 365d / 540d / 730d rankings.

- The reference date is the newest valid match date in the supplied analysis data, making reruns deterministic.
- Player eligibility still uses raw pressure deliveries and distinct matches (default: 100 deliveries and 10 matches). Recency does not let a small recent sample bypass these gates.
- Batter performance is weighted runs per 100 pressure deliveries.
- Bowler performance is weighted credited wickets per 100 pressure deliveries.
- Weighted rates are shrunk toward the corresponding weighted role-wide rate using a fixed 100 weighted-delivery-equivalent prior.
- The dashboard compares Top-10 membership overlap and rank movement across half-lives.

### Historical match-outcome model

The actual-outcome XGBoost benchmark is trained four times: unweighted, 365d, 540d and 730d. Its binary target is whether the second-innings batting side eventually won.

- All experiments use the same chronological match-level train/test split.
- Recency weights are calculated from the latest training match date only.
- Recency weights are applied to training rows only; held-out test rows receive identical match-balanced evaluation weights for every model.
- The 540-day model is saved as a provisional dashboard default. It is not assumed to be optimal.
- Compare out-of-time Brier score and log loss (lower is better), ROC-AUC (higher is better), and calibration. These measure match-outcome prediction, not psychological pressure.

Generated files:
- `outputs/player_rankings.csv` — full unweighted and decayed player rankings.
- `outputs/player_top10_comparison.csv` — eligible Top-10 rows across scoring methods and half-lives.
- `outputs/historical_win_probability_recency_comparison.csv` — holdout metrics for all four outcome-model training schemes.
- `outputs/metrics/historical_win_probability_metrics.json` — detailed metrics for the provisional 540-day model.
- `models/historical_win_probability_xgb_540d.json` — provisional 540-day outcome model.

**Interpretation:** recency can improve current relevance but can also increase variance by down-weighting older sample. Do not pick the half-life based only on a higher recent player rate. Consider out-of-time model performance, raw sample size, shrinkage and ranking stability together. The 540-day setting is a comparison candidate, not a proven winner.


**Chronological model leakage guard:** the historical win-probability model excludes the season-wide target percentile and season-baseline-filled recent run-rate/momentum features, because those were computed using all matches in a season and could indirectly expose future match information to earlier training rows. The outcome model uses only pre-delivery state variables plus match metadata available before play.

## Final Selection Lab — pressure consistency

The **Final Selection Lab** ranks repeatable high-pressure performance. It does not rank players by uplift over their own baseline.

### Performance measures

- **Batters:** runs per 100 balls faced in high-pressure chase deliveries. Wides are excluded from balls faced. Dismissals per 100 balls faced are displayed as a secondary indicator.
- **Bowlers:** dot-ball percentage on known legal deliveries. Rows with unresolved bye/legbye run values are excluded from the dot-ball denominator. Credited wickets per 100 legal balls are displayed separately because wicket counts are sparse and volatile; economy is not reported until score reconstruction is resolved.
- **Match consistency:** player performance is first aggregated within each high-pressure match. The system then calculates a recency- and exposure-weighted mean and standard deviation across matches.
- **Reliability adjustment:** the mean is shrunk toward the role-wide high-pressure rate using a five-effective-match prior. The consistency score subtracts a configurable number of standard errors from that shrunk mean. This rewards players with strong performance that repeats across matches and penalizes volatility or low effective samples.
- **Eligibility:** the default minimum is 100 measured pressure deliveries across at least 10 matches. The gates apply to the measured denominator (balls faced for batters; known legal deliveries for bowlers).
- **Consensus:** the final order is the median percentile across four PI definitions: baseline, RRR-heavy, momentum-heavy and stakes-heavy. A candidate must qualify under at least three. Synthetic-trained XGBoost is displayed as a diagnostic but excluded from consensus.

The dashboard supports no decay, 365-day, 540-day and 730-day half-lives, player-name search, method-level Top-10 and Top-20 overlap matrices, consensus rank movement, and Top-10 / Top-20 overlap across recency settings. The main sidebar's format and season filters define the player pool.

**Interpretation guardrail:** these are descriptive pressure-performance consistency rankings, not proof of psychological clutch ability. The bowling primary metric is dot-ball percentage rather than economy because unresolved bye/legbye values remain in the source data.
