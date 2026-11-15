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

**The dataset folder is intentionally gitignored.** Its README remains tracked, but CSVs will not be committed or pushed. Earlier project bundles may already have the same six files under `data/raw/`; the loader supports that legacy location too, and it is also gitignored. Keep the source data local unless the competition terms explicitly allow redistribution. See [dataset/README.md](dataset/README.md) for exact steps and an ignore check.

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



### 5. Weighted Reciprocal Rank Fusion (RRF) player shortlist

The ranking refresh and full analysis now generate an ensemble shortlist using weighted Reciprocal Rank Fusion. RRF aggregates the player orderings from the five available scoring methods:

RRF(p) = sum over methods m of [w_m / (60 + rank_m(p))]

The weights are deliberately unequal:

| Scoring method | RRF weight | Rationale |
|---|---:|---|
| PI baseline | 35% | Main transparent, pre-declared pressure definition |
| PI RRR-heavy | 25% | Alternative emphasising required-rate stress |
| PI momentum-heavy | 20% | Alternative emphasising recent scoring shortfall |
| PI stakes-heavy | 15% | Alternative emphasising match-stage importance |
| Synthetic-trained XGBoost | 5% | Low-weight sensitivity signal because its classes come from a synthetic teacher rule, not empirical pressure labels |

The four PI methods share the same underlying feature space and are sensitivity variants, not four independent experiments. Their weights are therefore not interpreted as independent votes. Synthetic XGBoost receives only 5%; the historical match-outcome model is not given a player-level vote because it predicts team chase outcomes, not an individual batter's or bowler's pressure performance.

**Candidate eligibility:** each model ranking is recomputed after applying the selected minimum pressure-delivery count, minimum match count and recent-activity window. A player must appear in at least three of the five eligible model lists by default. Missing model ranks contribute zero to the RRF score; model support and total supporting weight are reported separately. This prevents a high score in only one or two methods from creating a shortlist place.

The dashboard's **RRF ensemble shortlist** tab lets you change the role, sample gates, half-life, activity window, minimum model support and shortlist size. Treat the result as a candidate pool for human review—not an automatic final Top 10. RRF measures agreement between configured scoring methods, not causal clutch ability or improvement over a player's own baseline.

Generated files:
- outputs/player_rrf_ensemble.csv — RRF results for all available recency settings.
- outputs/player_rrf_top20.csv — first 20 eligible candidates per role and recency setting.

## Pull future code updates without replacing your local dataset

From your existing clone, first confirm the source CSV is ignored, then fast-forward to the latest code:

```bash
git status --short
git check-ignore dataset/ball_by_ball.csv
git pull --ff-only origin main
```

The `dataset/README.md` file is tracked, but all other files inside `dataset/` are ignored. Therefore your local source CSVs stay on your machine during a normal pull. Generated files under `outputs/`, `models/`, `data/processed/` and `data/synthetic/` are also ignored.

After pulling ranking-recency changes, refresh the player tables from the already processed delivery data and relaunch the dashboard:

```bash
python refresh_player_rankings.py
streamlit run dashboard/app.py
```

This standalone refresh uses `data/processed/real_pressure_comparison.csv.gz` and does **not** retrain the synthetic or historical outcome models. Run `python run_analysis.py` only when the processed delivery data is missing or you have replaced/updated the source CSVs.

Do **not** run `git clean -fdx` in this project: that command also deletes ignored files, including your local `dataset/` CSVs and generated analysis artifacts.


## Recency-weighted performance experiments

Recent performances are often more relevant to a current player-selection decision than older performances. Player rankings compare **no decay, 180-day, 365-day, 540-day and 730-day exponential half-lives** instead of hard-coding a single recency assumption. The 180-day setting is the new default for current player selection; longer half-lives remain available for sensitivity analysis.

For a match age of (a) days and half-life (H):

```text
recency_weight = 2 ** (-age_days / half_life_days)
```

At the half-life, a match receives half the weight of a match on the reference date. At twice the half-life, it receives one-quarter weight.

### Player rankings

For each pressure definition (PI baseline, three PI weight alternatives and synthetic-trained XGBoost), batters and bowlers receive an unweighted ranking plus 180d / 365d / 540d / 730d rankings.

- The reference date defaults to the current UTC date, not the newest match date inside the source data pack. This ensures an inactive player's last appearance continues to age even when the data pack itself is old. Pass an explicit reference date to reproduce a historical selection snapshot.
- Player eligibility still uses raw pressure deliveries and distinct matches (default: 100 deliveries and 10 matches). Recency does not let a small recent sample bypass these gates.
- Batter performance is weighted runs per 100 pressure deliveries.
- Bowler performance is weighted credited wickets per 100 pressure deliveries.
- Weighted rates are shrunk toward the corresponding weighted role-wide rate using a fixed 100 weighted-delivery-equivalent prior.
- The dashboard defaults to a 180-day half-life and offers a recent-pressure-appearance filter (last 365 days, last 730 days, or all history). It displays the last high-pressure appearance, days since that appearance, and pressure deliveries in the last 365 days.

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


## Rajasthan Royals Squad Intelligence

A separate RR-specific module analyses Rajasthan Royals match situations and builds role/phase candidate shortlists from the wider IPL player pool. It does not alter the competition's primary pressure Top-10 rankings.

### Data source and local setup

The ingestion script uses Cricsheet's IPL JSON archive:

- Download: https://cricsheet.org/downloads/ipl_json.zip
- Format documentation: https://cricsheet.org/format/json/
- Licence and attribution: https://cricsheet.org/licensing/

Download and build the local report for IPL seasons 2022–2026:

python scripts/build_rr_intelligence.py --download --start-year 2022 --end-year 2026 --xi-season 2026

The archive is saved to dataset/rr_intelligence/ipl_json.zip. It is local-only: dataset files are ignored by Git. Generated outputs are written under outputs/rr_intelligence/, which is also ignored. Do not upload the archive or generated match data unless the applicable licence and competition terms permit it.

### Analytical tracks

1. **RR match diagnostics:** filter matches involving Rajasthan Royals and reconstruct every delivery, including runs, extras, legal balls, wickets, target, score, required rate and remaining resources.
2. **Pressure points:** batter pressure uses a chase-state index based on required rate, wickets lost, time elapsed and recent scoring shortfall. Bowling pressure is a separate role-specific index that reflects scoring threat, wickets in hand, time elapsed and chase control. These are transparent heuristics, not psychological ground truth.
3. **Turning-point windows:** aggregate high-pressure deliveries by match and phase to locate periods where the chase was under strain or the bowling side was conceding control.
4. **RR phase gaps:** compare RR batting and bowling performance in Powerplay, Middle and Death phases against the rest of the selected IPL data.
5. **Player fit:** rank all-IPL player profiles within role and phase. Batters are assessed using strike rate, boundary rate, dismissal rate and pressure scoring; bowlers use economy, wickets, dots and pressure economy. The fit score combines player performance with the measured RR phase need.

### Generated files

- outputs/rr_intelligence/rr_match_summary.csv
- outputs/rr_intelligence/rr_pressure_points.csv
- outputs/rr_intelligence/rr_bowling_pressure_points.csv
- outputs/rr_intelligence/rr_pressure_windows.csv
- outputs/rr_intelligence/rr_phase_gaps.csv
- outputs/rr_intelligence/rr_player_fit_shortlist.csv
- outputs/rr_intelligence/rr_intelligence_report.md
- outputs/rr_intelligence/rr_deliveries.csv.gz and all_ipl_deliveries.csv.gz

The Streamlit dashboard contains an **RR squad intelligence** tab after the RRF shortlist.

### Interpretation limits

- Chase pressure can be calculated only where a valid target exists. DLS games without an explicit revised target are excluded from pressure scoring.
- First-innings deliveries remain useful for general role/phase player baselines but are not assigned a chase pressure score.
- The batter and bowler pressure formulas are separate heuristics. Their scores should not be interpreted as comparable psychological measurements.
- Candidate player scores are relative, within-role and within-phase performance percentiles; they are not absolute quality ratings.
- The candidate shortlist does not establish 2027 auction availability, retention status, overseas-slot fit, expected price or squad balance. Join it to an authoritative auction/retention list before making acquisition recommendations.
- Preserve Cricsheet attribution and comply with its ODC-By licence when redistributing derived data.


### PI / XGBoost phase selector and historical RR frequent XI

The RR build loads the saved synthetic_pressure_xgb.json model and applies the repository's PI feature definitions plus phase-specific PI weightings to Cricsheet-derived chase states.

- Batting pressure uses the selected phase PI and XGBoost chase score.
- Bowling pressure uses the inverse of the phase PI and XGBoost score, so a comfortable chase represents greater stress on the defending side.
- Powerplay uses baseline PI, Middle uses momentum-heavy PI, and Death uses RRR-situation-heavy PI.
- Players are ranked within role and phase using performance in model-identified high-pressure deliveries: batters by strike rate, boundary rate and dismissal avoidance; bowlers by economy, credited wickets and dot-ball rate.
- The phase decision score uses 80% PI performance percentile and 20% synthetic-XGBoost performance percentile. If a player lacks enough sample for one signal, available weights are renormalised and coverage is shown.
- Minimum delivery and match gates can be adjusted in the dashboard.

The PI/XGB phase selector & RR XI tab provides:
- A phase selector for Powerplay, Middle and Death, for batters and bowlers, across all IPL players or the historical RR frequent XI.
- The 11 players most frequently listed in RR playing XIs in IPL 2026 only.
- Six phase options for each frequent-XI player: batting and bowling in Powerplay, Middle and Death.

New outputs:
- outputs/rr_intelligence/rr_phase_player_rankings.csv
- outputs/rr_intelligence/rr_frequent_xi.csv
- outputs/rr_intelligence/rr_frequent_xi_phase_options.csv

Important: XGBoost is trained on synthetic teacher labels, so it is a supporting signal and not empirically validated psychological-pressure ground truth. The frequent XI membership and appearance counts are based on IPL 2026 only; the player phase-performance scores use the 2022–2026 IPL sample. This is not the official 2027 retained squad. Confirm current retention and auction availability independently.

## Final pressure rankings and RR XI replacement decisions

The latest decision layer makes two distinct outputs available in the dashboard's
**Final submission + RR XI decisions** tab.

### Competition submission: top 10 batters and top 10 bowlers

The final submission view runs weighted Reciprocal Rank Fusion (RRF) over the
existing pressure ranking methods. Default method weights remain:

| Ranking method | RRF weight |
|---|---:|
| PI baseline | 35% |
| PI RRR-heavy | 25% |
| PI momentum-heavy | 20% |
| PI stakes-heavy | 15% |
| Synthetic-trained XGBoost | 5% |

The final-submission tab defaults to 50 pressure deliveries, 5 pressure matches,
recent activity within 365 days, and support from at least three methods. A
100-delivery prior shrinks small samples toward the role-wide rate, and all four
PI variants plus XGBoost must be considered before the RRF support gate is applied.
The thresholds and recency horizon remain adjustable. The lower entry gate is
used because the saved 2022–2026 data had only 12 batters passing the stricter
100-delivery / 10-match gate; 50 / 5 produced a wider eligible set while retaining
shrinkage and cross-method agreement. RRF is a ranking-consensus mechanism: it
rewards players who rank consistently across model variants rather than treating
raw scores from different methods as directly interchangeable.

The player scoring logic is role-specific:

- **Batters:** evaluate high batting-side pressure (PI/XG score ≥60) and compare
  their shrunk runs per 100 pressure balls with their own low-pressure chase
  baseline (score <40). The selection score combines pressure scoring-rate
  percentile (75%) and personal pressure-uplift percentile (25%).
- **Bowlers:** evaluate defending-side pressure, i.e. the inverse chase state
  (batting-side pressure score ≤40). Use legal balls, credited wickets per 100
  pressure balls, economy rate, dot-ball rate, and uplift in wicket rate versus
  the player's own low-bowling-pressure baseline. The selection score weights
  wicket-rate percentile (45%), economy percentile (20%), dot rate (15%), and
  personal wicket-rate uplift (20%).
- **Bowling accounting:** wides and no-balls are included in runs charged to the
  bowler, but excluded from legal-ball denominators. Byes and leg-byes are not
  charged to the bowler; however, a legal delivery that produces byes/leg-byes
  is not a dot ball. Dot rate therefore uses total runs off the delivery, while
  economy uses only runs charged to the bowler. In the supplied IPL/T20I CSV,
  bye/leg-bye run reconstruction is incomplete for some rows, so global dot
  metrics involving those rows remain provisional; the Cricsheet RR source has
  complete delivery-run values.

The RRF weight assigned to synthetic-trained XGBoost is intentionally only 5%.
Its labels are a declared synthetic teacher rule; its holdout metrics do not
validate psychological pressure in real matches. The PI variants get 95% total
weight. Use the exported CSVs as the ranked list for manual review, then verify
player-name spelling and dropdown eligibility on the competition portal.

### Rajasthan Royals: current-XI pressure diagnosis and replacements

The RR decision tab uses the most frequently listed RR playing XI in **IPL 2026**
as the current/historical XI reference. It is not an official 2027 retention list.
For each XI member's observed role and each phase (Powerplay, Middle, Death), it:

1. Measures RR's phase-specific batting/bowling need against the rest of the IPL.
2. Compares the XI player's RR chase performance in high pressure against that
   same player's lower-pressure chase baseline. Batting evidence includes strike
   rate, boundary rate and dismissal rate; bowling evidence includes economy,
   wickets per 100 legal balls and dot-ball rate.
3. Joins the player's phase performance to the all-IPL PI/XGBoost phase ranking
   and labels the finding as a measured pressure weakness, a team phase need with
   limited individual RR sample, or no strong weakness signal.
4. Retrieves a same-role, same-phase candidate from the wider IPL pool, excludes
   current XI members and ranks candidates using 85% phase PI/XGBoost decision
   score plus 15% general role/phase fit.

Phase-specific PI settings are selected by the cricket task:

| Phase | PI variant | Why |
|---|---|---|
| Powerplay | Baseline PI | Balanced read of early chase context and wickets |
| Middle | Momentum-heavy PI | Emphasises a scoring-rate shortfall and ability to control the chase |
| Death | RRR-heavy PI | Emphasises the required scoring rate and limited balls remaining |

The phase player score remains 80% PI / 20% synthetic-trained XGBoost. The bowler
uses the inverse of the batting-side pressure state. RR replacement candidates
normally need at least 20 PI-pressure deliveries across 3 matches; candidates that
only pass a relaxed phase-exposure gate are explicitly labelled.

**Final-over lens:** separately from the broader Death phase, the selector measures
the last six balls of an active chase (balls_remaining <= 6). RR XI batters are
compared on last-over strike rate and dismissal/boundary rates against their own
lower-pressure last-over baseline; bowlers are compared on economy, credited
wickets and dot rate. A final-over weakness is allowed to affect the XI finding
only with at least 8 pressure balls across 2 matches. Death-phase replacement
candidates can receive an additional 15% last-over fit component when they clear
the same sample gate; otherwise the selector falls back to the normal Death-phase
score. This avoids elevating a candidate on a one- or two-ball anecdote.

Auction/retention availability, overseas slots, budget, and final squad balance
are not inferred.

### Rebuild the new outputs

From the repository root, with the six authorised CSV files in `dataset/`:

```bash
python run_analysis.py
python scripts/build_rr_intelligence.py --download --start-year 2022 --end-year 2026 --xi-season 2026
streamlit run dashboard/app.py
```

The RR build writes two additional outputs:

- `outputs/rr_intelligence/rr_xi_pressure_weakness_review.csv`
- `outputs/rr_intelligence/rr_xi_replacement_shortlist.csv`

The Final Top 10 tab reads the regenerated `outputs/player_rankings.csv`.
Generated data, archives, outputs and model files remain local/ignored; push code
and documentation only. The main data-quality caveat remains: the supplied
ball-by-ball source encodes byes/legbyes in a categorical extras field without
unambiguous numeric run counts. Reconcile affected score reconstructions before
making definitive claims about pressure failures.


## Final ranking eligibility

For the submitted IPL + T20I Top 10 lists, players mapped to the Pakistan national team in
`teams.csv` and `competition_team_player_mapping.csv` are excluded from candidate rankings.
The source deliveries are retained for match-state construction and for calculating other
players' performance; only the candidate's own final ranking rows are removed. The analysis
fails with a clear error if the source team metadata does not contain a Pakistan national-team
record, rather than silently generating a list without applying the configured eligibility rule.
