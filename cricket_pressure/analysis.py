from __future__ import annotations

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from .features import load_source_tables, build_delivery_states
from .pressure import DEFAULT_WEIGHTS, add_pressure_features, compute_pressure_index
from .synthetic import (
    CLASS_NAMES, FEATURE_COLUMNS, generate_synthetic_scenarios,
    train_synthetic_xgb, score_with_xgb,
)
from .player_rankings import (
    build_player_rankings, compare_top10_rankings,
    build_weighted_rrf_ensemble, compare_top_rrf_ensemble,
)
from .outcome import train_historical_win_probability
import re

ALT_WEIGHTS = {
    "baseline": DEFAULT_WEIGHTS,
    "rrr_situation_heavy": {
        "rrr_pressure": .35, "wicket_pressure": .20, "time_urgency": .15,
        "momentum_pressure": .10, "death_over_pressure": .08,
        "target_difficulty": .07, "match_stakes": .05,
    },
    "momentum_heavy": {
        "rrr_pressure": .20, "wicket_pressure": .16, "time_urgency": .12,
        "momentum_pressure": .25, "death_over_pressure": .10,
        "target_difficulty": .08, "match_stakes": .09,
    },
    "stakes_heavy": {
        "rrr_pressure": .20, "wicket_pressure": .16, "time_urgency": .12,
        "momentum_pressure": .12, "death_over_pressure": .08,
        "target_difficulty": .12, "match_stakes": .20,
    },
}


def _ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "outputs": root / "outputs",
        "figures": root / "outputs" / "figures",
        "metrics": root / "outputs" / "metrics",
        "processed": root / "data" / "processed",
        "synthetic": root / "data" / "synthetic",
        "models": root / "models",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _build_scenario_lab(model) -> pd.DataFrame:
    scenarios = pd.DataFrame([
        ["Early, comfortable chase", 6.5, 9, 102, 120, 8.0, .35, .25, 0.0, 0],
        ["Early, high required rate", 14.0, 9, 102, 120, 8.0, .85, .25, 0.0, 0],
        ["Death overs, manageable chase", 8.0, 6, 18, 120, 9.0, .55, .25, .4, 0],
        ["Death overs, high required rate", 17.0, 5, 18, 120, 9.0, .90, .25, .4, 0],
        ["Death overs, only two wickets", 13.0, 2, 12, 120, 8.0, .75, .25, .6, 0],
        ["High target, early overs", 10.0, 9, 96, 120, 9.0, .98, .25, 0.0, 0],
        ["Knockout, otherwise moderate chase", 9.0, 7, 48, 120, 8.5, .55, .95, 0.0, 0],
        ["Short-format chase, tight finish", 12.0, 4, 12, 60, 10.0, .80, .90, .6, 1],
    ], columns=["scenario", *FEATURE_COLUMNS])
    # Add PI component scores using exactly the same transformations as real data.
    enriched = scenarios.copy()
    enriched["rrr_pressure"] = (enriched["required_run_rate"] / 12).clip(0, 1)
    enriched["wicket_pressure"] = (1 - enriched["wickets_remaining"] / 10).clip(0, 1)
    enriched["time_urgency"] = (1 - enriched["balls_remaining"] / enriched["scheduled_balls"]).clip(0, 1)
    enriched["momentum_pressure"] = ((enriched["required_run_rate"] - enriched["recent_run_rate_12"]).clip(lower=0) / 12).clip(0, 1)
    enriched["death_over_pressure"] = (1 - enriched["balls_remaining"] / 30).clip(0, 1)
    enriched["target_difficulty"] = enriched["target_difficulty"].clip(0, 1)
    enriched["match_stakes"] = enriched["match_stakes"].clip(0, 1)
    for col in DEFAULT_WEIGHTS:
        if col not in enriched:
            raise KeyError(col)
    enriched["pressure_index"] = sum(enriched[k] * w for k, w in DEFAULT_WEIGHTS.items()) * 100
    enriched = score_with_xgb(model, enriched)
    enriched["pi_category"] = pd.cut(enriched["pressure_index"], [-.001, 30, 60, 80, 100], labels=CLASS_NAMES)
    return enriched


def _player_summary(real: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if real.empty:
        return pd.DataFrame()
    batter_rows = real.copy()
    batter_rows["dismissed_striker"] = (
        pd.to_numeric(batter_rows["dismissed_player_id"], errors="coerce") ==
        pd.to_numeric(batter_rows["striker_id"], errors="coerce")
    ).astype(int)
    bat = batter_rows.groupby(["striker_id", "striker_name"], dropna=False).agg(
        role_deliveries=("match_id", "size"),
        matches=("match_id", "nunique"),
        batter_runs=("runs", "sum"),
        pi_mean=("pressure_index", "mean"),
        xgb_mean=("xgb_pressure_score", "mean"),
        high_pi_deliveries=("pressure_index", lambda s: int((s >= 60).sum())),
        dismissals_of_batter=("dismissed_striker", "sum"),
    ).reset_index().rename(columns={"striker_id": "player_id", "striker_name": "player_name"})
    bat["role"] = "Batter"
    bat["runs_per_100_pressure_deliveries"] = 100 * bat["batter_runs"] / bat["role_deliveries"].replace(0, np.nan)
    frames.append(bat)

    credited = {"caught", "bowled", "lbw", "stump", "hit wicket"}
    b = real.copy()
    b["credited_bowler_wicket"] = (
        b["is_wicket"].fillna(False).astype(bool) &
        b["dismissal_type"].fillna("").astype(str).str.lower().isin(credited)
    ).astype(int)
    bowl = b.groupby(["bowler_id", "bowler_name"], dropna=False).agg(
        role_deliveries=("match_id", "size"),
        matches=("match_id", "nunique"),
        wickets=("credited_bowler_wicket", "sum"),
        provisional_runs_conceded=("provisional_total_runs", "sum"),
        pi_mean=("pressure_index", "mean"),
        xgb_mean=("xgb_pressure_score", "mean"),
        high_pi_deliveries=("pressure_index", lambda s: int((s >= 60).sum())),
    ).reset_index().rename(columns={"bowler_id": "player_id", "bowler_name": "player_name"})
    bowl["role"] = "Bowler"
    bowl["wickets_per_100_pressure_deliveries"] = 100 * bowl["wickets"] / bowl["role_deliveries"].replace(0, np.nan)
    bowl["provisional_runs_per_6_deliveries"] = 6 * bowl["provisional_runs_conceded"] / bowl["role_deliveries"].replace(0, np.nan)
    frames.append(bowl)
    return pd.concat(frames, ignore_index=True, sort=False)


def _plot_outputs(synthetic_test: pd.DataFrame, metrics: dict, importance: pd.DataFrame,
                  real: pd.DataFrame, sensitivity: pd.DataFrame, dirs: dict[str, Path]) -> None:
    # Synthetic holdout confusion matrix.
    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), CLASS_NAMES, rotation=20)
    ax.set_yticks(range(4), CLASS_NAMES)
    ax.set_xlabel("Predicted synthetic class")
    ax.set_ylabel("Synthetic teacher class")
    ax.set_title("XGBoost — synthetic holdout confusion matrix")
    for i in range(4):
        for j in range(4): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Scenarios")
    fig.tight_layout(); fig.savefig(dirs["figures"] / "synthetic_confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    imp = importance.sort_values("importance")
    ax.barh(imp["feature"], imp["importance"])
    ax.set_title("XGBoost feature importance (synthetic task)")
    ax.set_xlabel("Gain-based importance")
    fig.tight_layout(); fig.savefig(dirs["figures"] / "xgb_feature_importance.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sample = synthetic_test.sample(min(5000, len(synthetic_test)), random_state=12)
    ax.scatter(sample["synthetic_teacher_score"], sample["xgb_predicted_score"], alpha=.25, s=12)
    ax.plot([0, 100], [0, 100], linestyle="--")
    ax.set(xlabel="Synthetic teacher score", ylabel="XGBoost expected class score",
           title="XGBoost approximation of synthetic scenario labels")
    fig.tight_layout(); fig.savefig(dirs["figures"] / "synthetic_score_fit.png", dpi=150); plt.close(fig)

    if not real.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        sample = real.sample(min(6000, len(real)), random_state=12)
        ax.scatter(sample["pressure_index"], sample["xgb_pressure_score"], alpha=.25, s=12)
        ax.set(xlabel="PI heuristic score", ylabel="XGBoost synthetic-trained score",
               title="Historical match states: PI vs synthetic-trained XGBoost")
        fig.tight_layout(); fig.savefig(dirs["figures"] / "pi_vs_xgb_scatter.png", dpi=150); plt.close(fig)

        cat = pd.concat([
            real["pressure_category"].astype(str).value_counts().rename("PI").reindex(CLASS_NAMES, fill_value=0),
            real["xgb_pressure_class"].astype(str).value_counts().rename("XGBoost").reindex(CLASS_NAMES, fill_value=0),
        ], axis=1)
        fig, ax = plt.subplots(figsize=(8, 5))
        cat.plot(kind="bar", ax=ax)
        ax.set(title="Historical chase-state category counts", ylabel="Deliveries", xlabel="Pressure class")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout(); fig.savefig(dirs["figures"] / "pi_vs_xgb_class_counts.png", dpi=150); plt.close(fig)

    if not sensitivity.empty:
        corr = sensitivity.set_index("scenario").drop(columns=["mean_score", "median_score"], errors="ignore").T.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(corr, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(corr)), corr.columns, rotation=25, ha="right")
        ax.set_yticks(range(len(corr)), corr.index)
        ax.set_title("Spearman correlation between PI weight scenarios")
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", color="white" if corr.iloc[i,j] < .55 else "black")
        fig.colorbar(im, ax=ax, label="Spearman ρ")
        fig.tight_layout(); fig.savefig(dirs["figures"] / "weight_sensitivity_heatmap.png", dpi=150); plt.close(fig)


def _build_score_audit(states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build innings checks and a conservative run-margin reconciliation audit.

    The supplied pack does not contain independent official innings totals.
    This audit therefore checks internal integrity and compares reconstructed
    margins with the textual result only for non-DLS run wins with no unresolved
    bye/legbye deliveries. It never silently labels provisional totals as official.
    """
    cols = [
        "match_id", "innings", "provisional_innings_total", "legal_balls_total",
        "unresolved_extra_deliveries", "wickets_total", "scheduled_balls_for_audit",
    ]
    innings = states[cols].drop_duplicates(["match_id", "innings"]).copy()
    innings["legal_balls_over_schedule"] = (
        innings["legal_balls_total"] > innings["scheduled_balls_for_audit"]
    )
    innings["wickets_over_limit"] = innings["wickets_total"] > 10

    meta_cols = [c for c in ["match_id", "match_date", "comp_name", "match_format", "season", "result"] if c in states]
    meta = states[meta_cols].drop_duplicates("match_id").set_index("match_id")
    audit = meta.copy()
    for inn in (1, 2):
        part = innings.loc[innings["innings"].eq(inn)].set_index("match_id")
        audit[f"provisional_runs_inn{inn}"] = part["provisional_innings_total"]
        audit[f"legal_balls_inn{inn}"] = part["legal_balls_total"]
        audit[f"scheduled_balls_inn{inn}"] = part["scheduled_balls_for_audit"]
        audit[f"unresolved_extras_inn{inn}"] = part["unresolved_extra_deliveries"]
        audit[f"wickets_inn{inn}"] = part["wickets_total"]

    audit = audit.reset_index()
    audit["unresolved_extra_deliveries"] = (
        audit[["unresolved_extras_inn1", "unresolved_extras_inn2"]].fillna(0).sum(axis=1)
    )
    audit["reconstructed_run_margin"] = (
        pd.to_numeric(audit["provisional_runs_inn1"], errors="coerce")
        - pd.to_numeric(audit["provisional_runs_inn2"], errors="coerce")
    )
    result_text = audit.get("result", pd.Series("", index=audit.index)).fillna("").astype(str)
    audit["reported_run_margin"] = pd.to_numeric(
        result_text.str.extract(r"(?i)won by\s+(\d+)\s+runs?")[0], errors="coerce"
    )
    audit["is_dls_result"] = result_text.str.contains(r"(?i)(D/L|DLS|Duckworth)", regex=True)
    audit["run_margin_audit_eligible"] = (
        audit["reported_run_margin"].notna()
        & ~audit["is_dls_result"]
        & audit["unresolved_extra_deliveries"].eq(0)
        & audit["provisional_runs_inn1"].notna()
        & audit["provisional_runs_inn2"].notna()
    )
    audit["run_margin_abs_error"] = (
        audit["reconstructed_run_margin"] - audit["reported_run_margin"]
    ).abs().where(audit["run_margin_audit_eligible"])
    audit["run_margin_audit_status"] = np.select(
        [
            audit["is_dls_result"],
            audit["unresolved_extra_deliveries"].gt(0),
            audit["reported_run_margin"].isna(),
            audit["run_margin_abs_error"].eq(0),
        ],
        [
            "not comparable: DLS/DL result",
            "provisional: unresolved bye/legbye runs",
            "not a run-margin result",
            "consistent with reported margin",
        ],
        default="mismatch: investigate score reconstruction",
    )
    audit["innings_ball_integrity_issue"] = (
        audit["legal_balls_inn1"].fillna(0).gt(audit["scheduled_balls_inn1"].fillna(0))
        | audit["legal_balls_inn2"].fillna(0).gt(audit["scheduled_balls_inn2"].fillna(0))
    )
    audit["wicket_integrity_issue"] = (
        audit[["wickets_inn1", "wickets_inn2"]].fillna(0).gt(10).any(axis=1)
    )
    return innings, audit


def _build_match_disagreements(real: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise where historical PI and synthetic-trained XGB diverge."""
    d = real.copy()
    d["pi_xgb_gap"] = d["xgb_pressure_score"] - d["pressure_index"]
    d["absolute_pi_xgb_gap"] = d["pi_xgb_gap"].abs()
    detail_cols = [
        c for c in [
            "match_id", "match_date", "comp_name", "match_format", "season",
            "innings_ball_num", "overid", "score_before", "runs_required",
            "required_run_rate", "wickets_remaining", "balls_remaining",
            "pressure_index", "xgb_pressure_score", "xgb_pressure_class",
            "pi_xgb_gap", "absolute_pi_xgb_gap",
        ] if c in d.columns
    ]
    details = d.sort_values("absolute_pi_xgb_gap", ascending=False)[detail_cols].head(500).copy()
    keys = [c for c in ["match_id", "match_date", "comp_name", "match_format", "season"] if c in d.columns]
    summary = d.groupby(keys, dropna=False).agg(
        chase_deliveries=("match_id", "size"),
        pi_mean=("pressure_index", "mean"),
        xgb_mean=("xgb_pressure_score", "mean"),
        mean_absolute_gap=("absolute_pi_xgb_gap", "mean"),
        max_absolute_gap=("absolute_pi_xgb_gap", "max"),
        pi_high_deliveries=("pressure_index", lambda s: int((s >= 60).sum())),
        xgb_high_deliveries=("xgb_pressure_score", lambda s: int((s >= 60).sum())),
    ).reset_index().sort_values(["mean_absolute_gap", "max_absolute_gap"], ascending=False)
    return summary, details


def _plot_extra_diagnostics(
    calibration: pd.DataFrame,
    disagreements: pd.DataFrame,
    rankings: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    if not calibration.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.plot(
            calibration["mean_predicted_probability"],
            calibration["observed_win_rate"],
            marker="o", label="Historical model",
        )
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted win probability",
               ylabel="Observed batting-team win rate", title="Historical outcome model calibration")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(dirs["figures"] / "historical_win_probability_calibration.png", dpi=150)
        plt.close(fig)
    if not disagreements.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(disagreements["mean_absolute_gap"].dropna(), bins=35)
        ax.set(xlabel="Mean absolute PI–XGBoost gap per match", ylabel="Matches",
               title="Match-level pressure-model disagreement")
        ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(dirs["figures"] / "match_disagreement_distribution.png", dpi=150)
        plt.close(fig)
    if not rankings.empty:
        top = rankings.loc[rankings["eligible"] & rankings["rank"].le(10)].copy()
        for role in ["Batter", "Bowler"]:
            role_top = top.loc[top["role"].eq(role)]
            if role_top.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot = role_top.pivot_table(
                index="player_name", columns="scoring_method", values="rank", aggfunc="min"
            )
            pivot.plot(kind="barh", ax=ax)
            ax.invert_yaxis()
            ax.set(title=f"Top-10 rank positions by pressure model — {role}",
                   xlabel="Rank (lower is better)", ylabel="Player")
            ax.grid(axis="x", alpha=.2); fig.tight_layout()
            fig.savefig(dirs["figures"] / f"top10_rank_comparison_{role.lower()}.png", dpi=150)
            plt.close(fig)


def run_analysis(project_root: str | Path, n_synthetic: int = 40000, seed: int = 42) -> dict:
    root = Path(project_root)
    dirs = _ensure_dirs(root)
    # The current layout is dataset/. Older local bundles placed the same six
    # authorised source files in data/raw/. Support both without moving or
    # rewriting the user's local data.
    source_dir = root / "dataset"
    legacy_source_dir = root / "data" / "raw"
    required_files = (
        "ball_by_ball.csv", "matches.csv", "competitions.csv",
        "teams.csv", "playing_eleven.csv", "competition_team_player_mapping.csv",
    )

    def has_complete_source_pack(folder: Path) -> bool:
        return all(
            (folder / filename).exists() or (folder / f"{filename}.gz").exists()
            for filename in required_files
        )

    if not has_complete_source_pack(source_dir) and has_complete_source_pack(legacy_source_dir):
        source_dir = legacy_source_dir
    tables = load_source_tables(source_dir)
    states = build_delivery_states(tables)
    enriched = add_pressure_features(states)
    pi_scored = compute_pressure_index(enriched)
    real = pi_scored.loc[pi_scored["pressure_eligible"]].copy()

    synthetic = generate_synthetic_scenarios(n=n_synthetic, seed=seed)
    model, synthetic_metrics, synthetic_test, extras = train_synthetic_xgb(synthetic, random_state=seed)
    real = score_with_xgb(model, real)
    real["pi_category"] = real["pressure_category"].astype(str)

    # Weight sensitivity is calculated on identical pre-delivery states.
    sensitivity_rows = []
    score_cols = {}
    for name, weights in ALT_WEIGHTS.items():
        score = sum(real[k].fillna(0) * w for k, w in weights.items()) * 100
        score_cols[name] = score
        sensitivity_rows.append({"scenario": name, "mean_score": float(score.mean()), "median_score": float(score.median())})
    sensitivity = pd.DataFrame(sensitivity_rows)
    for name, score in score_cols.items():
        real[f"pi_{name}"] = score
    baseline = score_cols["baseline"]
    sens_pair_rows = []
    for name, score in score_cols.items():
        sens_pair_rows.append({"scenario": name, "spearman_vs_baseline": float(spearmanr(baseline, score).statistic)})
    sens_pairs = pd.DataFrame(sens_pair_rows)
    sensitivity = sensitivity.merge(sens_pairs, on="scenario", how="left")

    # Summary comparison; no empirical ground-truth pressure labels exist in the pack.
    real_corr = float(real[["pressure_index", "xgb_pressure_score"]].corr(method="spearman").iloc[0, 1])
    real_pearson = float(real[["pressure_index", "xgb_pressure_score"]].corr().iloc[0, 1])
    real["pressure_category"] = real["pressure_index"].pipe(
        lambda s: pd.cut(s, [-.001, 30, 60, 80, 100], labels=CLASS_NAMES)
    )
    class_comparison = pd.crosstab(real["pressure_category"], real["xgb_pressure_class"], dropna=False)
    # Player rankings compare high-pressure performance under several PI weightings and XGBoost.
    player_comparison = _player_summary(real)  # full-sample exposure reference; descriptive only
    player_rankings = build_player_rankings(real)
    player_top10 = compare_top10_rankings(player_rankings, top_n=10)
    player_rrf = build_weighted_rrf_ensemble(player_rankings, half_life=None)
    player_rrf_top20 = compare_top_rrf_ensemble(player_rrf, top_n=20)
    match_disagreements, delivery_disagreements = _build_match_disagreements(real)
    innings_validation, score_audit = _build_score_audit(states)

    # Historical outcome models: unweighted baseline plus three recency half-lives.
    # The chronological held-out period remains identically weighted for fair comparison.
    outcome_experiments = {}
    for half_life in (None, 365, 540, 730):
        outcome_experiments[half_life] = train_historical_win_probability(
            pi_scored, random_state=seed, recency_half_life_days=half_life
        )
    # 540 days is the provisional dashboard default, not a pre-declared winner.
    outcome_model, outcome_metrics, outcome_predictions, outcome_calibration, outcome_importance = (
        outcome_experiments[540]
    )
    outcome_comparison = pd.DataFrame([
        {
            "half_life": "Unweighted" if half_life is None else f"{half_life} days",
            "half_life_days": 0 if half_life is None else half_life,
            "train_matches": result[1]["train_matches"],
            "test_matches": result[1]["test_matches"],
            "accuracy_match_weighted": result[1]["accuracy_match_weighted"],
            "roc_auc_match_weighted": result[1]["roc_auc_match_weighted"],
            "brier_score_match_weighted": result[1]["brier_score_match_weighted"],
            "baseline_brier_score_match_weighted": result[1]["baseline_brier_score_match_weighted"],
            "log_loss_match_weighted": result[1]["log_loss_match_weighted"],
            "cutoff_date_utc": result[1]["cutoff_date_utc"],
            "recency_reference_date_utc": result[1]["recency_reference_date_utc"],
        }
        for half_life, result in outcome_experiments.items()
    ])
    outcome_calibration = outcome_calibration.copy()
    outcome_calibration["training_half_life"] = "540 days"
    match_summary = real.groupby(["match_format", "season"], dropna=False).agg(
        pressure_deliveries=("match_id", "size"),
        matches=("match_id", "nunique"),
        pi_mean=("pressure_index", "mean"),
        xgb_mean=("xgb_pressure_score", "mean"),
        pi_median=("pressure_index", "median"),
        xgb_median=("xgb_pressure_score", "median"),
    ).reset_index()
    scenario_lab = _build_scenario_lab(model)

    synthetic_metrics_out = {k: v for k, v in synthetic_metrics.items() if k not in {"xgb_classification_report", "pi_classification_report"}}
    synthetic_metrics_out["real_state_count"] = int(len(real))
    synthetic_metrics_out["real_match_count"] = int(real["match_id"].nunique())
    synthetic_metrics_out["real_pi_xgb_spearman"] = real_corr
    synthetic_metrics_out["real_pi_xgb_pearson"] = real_pearson
    synthetic_metrics_out["interpretation"] = (
        "Synthetic holdout metrics measure how well XGBoost learns the deliberately generated synthetic label rule. "
        "They do not validate real-world psychological pressure. Historical PI-vs-XGBoost correlation measures agreement only."
    )
    synthetic_metrics_out["source_data_caveat"] = (
        f"{int(real['unresolved_bye_legbye'].sum()):,} unresolved bye/legbye deliveries occur in the pressure-eligible chase subset; "
        "innings totals and chase states remain provisional until reconciled with official scorecards."
    )

    synthetic.to_csv(dirs["synthetic"] / "synthetic_pressure_scenarios.csv", index=False)
    synthetic_test.to_csv(dirs["synthetic"] / "synthetic_test_predictions.csv", index=False)
    real.to_csv(dirs["processed"] / "real_pressure_comparison.csv.gz", index=False, compression="gzip")
    player_comparison.to_csv(dirs["outputs"] / "player_pressure_comparison.csv", index=False)
    match_summary.to_csv(dirs["outputs"] / "match_format_season_summary.csv", index=False)
    class_comparison.to_csv(dirs["outputs"] / "pi_xgb_class_crosstab.csv")
    sensitivity.to_csv(dirs["outputs"] / "weight_sensitivity.csv", index=False)
    scenario_lab.to_csv(dirs["outputs"] / "scenario_lab.csv", index=False)
    extras["importance"].to_csv(dirs["outputs"] / "xgb_feature_importance.csv", index=False)
    player_rankings.to_csv(dirs["outputs"] / "player_rankings.csv", index=False)
    player_top10.to_csv(dirs["outputs"] / "player_top10_comparison.csv", index=False)
    player_rrf.to_csv(dirs["outputs"] / "player_rrf_ensemble.csv", index=False)
    player_rrf_top20.to_csv(dirs["outputs"] / "player_rrf_top20.csv", index=False)
    match_disagreements.to_csv(dirs["outputs"] / "match_disagreement_summary.csv", index=False)
    delivery_disagreements.to_csv(dirs["outputs"] / "delivery_disagreements.csv", index=False)
    innings_validation.to_csv(dirs["outputs"] / "innings_validation.csv", index=False)
    score_audit.to_csv(dirs["outputs"] / "score_reconciliation_audit.csv", index=False)
    outcome_predictions.to_csv(dirs["processed"] / "historical_win_probability_predictions.csv.gz", index=False, compression="gzip")
    outcome_calibration.to_csv(dirs["outputs"] / "historical_win_probability_calibration.csv", index=False)
    outcome_comparison.to_csv(dirs["outputs"] / "historical_win_probability_recency_comparison.csv", index=False)
    outcome_importance.to_csv(dirs["outputs"] / "historical_win_probability_feature_importance.csv", index=False)
    np.savetxt(dirs["metrics"] / "synthetic_xgb_confusion_matrix.csv",
               np.asarray(synthetic_metrics["confusion_matrix"], dtype=int), delimiter=",", fmt="%d")
    with open(dirs["metrics"] / "historical_win_probability_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(outcome_metrics), f, indent=2)
    with open(dirs["metrics"] / "experiment_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(synthetic_metrics_out), f, indent=2)
    with open(dirs["metrics"] / "synthetic_xgb_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(synthetic_metrics["xgb_classification_report"]), f, indent=2)
    with open(dirs["metrics"] / "synthetic_pi_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(synthetic_metrics["pi_classification_report"]), f, indent=2)
    model.save_model(dirs["models"] / "synthetic_pressure_xgb.json")
    outcome_model.save_model(dirs["models"] / "historical_win_probability_xgb_540d.json")
    _plot_outputs(synthetic_test, synthetic_metrics, extras["importance"], real, sensitivity, dirs)
    _plot_extra_diagnostics(outcome_calibration, match_disagreements, player_rankings, dirs)

    summary = {
        "selected_matches": int(states["match_id"].nunique()),
        "selected_deliveries_innings_1_2": int(len(states)),
        "chase_pressure_deliveries": int(len(real)),
        "chase_matches": int(real["match_id"].nunique()),
        "unresolved_bye_legbye_deliveries": int(states["unresolved_bye_legbye"].sum()),
        "synthetic_xgb_accuracy": synthetic_metrics["xgb_accuracy"],
        "synthetic_xgb_macro_f1": synthetic_metrics["xgb_macro_f1"],
        "synthetic_pi_accuracy": synthetic_metrics["pi_accuracy"],
        "synthetic_pi_macro_f1": synthetic_metrics["pi_macro_f1"],
        "real_pi_xgb_spearman": real_corr,
        "real_pi_xgb_pearson": real_pearson,
        "mean_pi": float(real["pressure_index"].mean()),
        "mean_xgb_synthetic_score": float(real["xgb_pressure_score"].mean()),
        "score_audit_run_margin_comparable_matches": int(score_audit["run_margin_audit_eligible"].sum()),
        "score_audit_run_margin_consistent_matches": int(score_audit["run_margin_audit_status"].eq("consistent with reported margin").sum()),
        "score_audit_unresolved_extra_matches": int(score_audit["unresolved_extra_deliveries"].gt(0).sum()),
        "innings_over_schedule_count": int(innings_validation["legal_balls_over_schedule"].sum()),
        "historical_win_probability": outcome_metrics,
    }
    with open(dirs["metrics"] / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)
    print(json.dumps(summary, indent=2))
    return {
        "states": states, "pi_scored": pi_scored, "real": real,
        "synthetic": synthetic, "synthetic_test": synthetic_test,
        "synthetic_metrics": synthetic_metrics, "importance": extras["importance"],
        "sensitivity": sensitivity, "class_comparison": class_comparison,
        "players": player_comparison, "player_rankings": player_rankings,
        "player_top10": player_top10, "scenario_lab": scenario_lab,
        "match_disagreements": match_disagreements, "delivery_disagreements": delivery_disagreements,
        "innings_validation": innings_validation, "score_audit": score_audit,
        "outcome_metrics": outcome_metrics, "outcome_comparison": outcome_comparison,
        "outcome_predictions": outcome_predictions, "outcome_calibration": outcome_calibration,
        "summary": summary,
    }