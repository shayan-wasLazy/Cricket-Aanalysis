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


def run_analysis(project_root: str | Path, n_synthetic: int = 40000, seed: int = 42) -> dict:
    root = Path(project_root)
    dirs = _ensure_dirs(root)
    tables = load_source_tables(root / "data" / "raw")
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
    player_comparison = _player_summary(real)
    match_summary = real.groupby(["match_format", "season"], dropna=False).agg(
        pressure_deliveries=("match_id", "size"),
        matches=("match_id", "nunique"),
        pi_mean=("pressure_index", "mean"),
        xgb_mean=("xgb_pressure_score", "mean"),
        pi_median=("pressure_index", "median"),
        xgb_median=("xgb_pressure_score", "median"),
    ).reset_index()
    scenario_lab = _build_scenario_lab(model)

    synthetic_metrics_out = {k: v for k, v in synthetic_metrics.items() if k not in {"classification_report", "confusion_matrix"}}
    synthetic_metrics_out["real_state_count"] = int(len(real))
    synthetic_metrics_out["real_match_count"] = int(real["match_id"].nunique())
    synthetic_metrics_out["real_pi_xgb_spearman"] = real_corr
    synthetic_metrics_out["real_pi_xgb_pearson"] = real_pearson
    synthetic_metrics_out["interpretation"] = (
        "Synthetic holdout metrics measure how well XGBoost learns the deliberately generated synthetic label rule. "
        "They do not validate real-world psychological pressure. Historical PI-vs-XGBoost correlation measures agreement only."
    )
    synthetic_metrics_out["source_data_caveat"] = (
        f"{int(real['unresolved_bye_legbye'].sum()):,} unresolved bye/legbye deliveries occur in the selected pre-delivery dataset; "
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
    with open(dirs["metrics"] / "experiment_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(synthetic_metrics_out), f, indent=2)
    with open(dirs["metrics"] / "synthetic_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(synthetic_metrics["classification_report"]), f, indent=2)
    model.save_model(dirs["models"] / "synthetic_pressure_xgb.json")
    _plot_outputs(synthetic_test, synthetic_metrics, extras["importance"], real, sensitivity, dirs)

    summary = {
        "selected_matches": int(states["match_id"].nunique()),
        "selected_deliveries_innings_1_2": int(len(states)),
        "chase_pressure_deliveries": int(len(real)),
        "chase_matches": int(real["match_id"].nunique()),
        "unresolved_bye_legbye_deliveries": int(states["unresolved_bye_legbye"].sum()),
        "synthetic_accuracy": synthetic_metrics["accuracy"],
        "synthetic_macro_f1": synthetic_metrics["macro_f1"],
        "real_pi_xgb_spearman": real_corr,
        "real_pi_xgb_pearson": real_pearson,
        "mean_pi": float(real["pressure_index"].mean()),
        "mean_xgb_synthetic_score": float(real["xgb_pressure_score"].mean()),
    }
    with open(dirs["metrics"] / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)
    print(json.dumps(summary, indent=2))
    return {
        "states": states, "pi_scored": pi_scored, "real": real,
        "synthetic": synthetic, "synthetic_test": synthetic_test,
        "synthetic_metrics": synthetic_metrics, "importance": extras["importance"],
        "sensitivity": sensitivity, "class_comparison": class_comparison,
        "players": player_comparison, "scenario_lab": scenario_lab,
        "summary": summary,
    }