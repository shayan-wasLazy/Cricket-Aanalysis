from __future__ import annotations

import numpy as np
import pandas as pd

# These are initial eligibility gates, not claims that 100 balls / 10 matches
# are universally sufficient. The dashboard lets the analyst change them.
DEFAULT_MIN_DELIVERIES = 100
DEFAULT_MIN_MATCHES = 10
DEFAULT_PRIOR_DELIVERIES = 100

SCORE_METHODS = {
    "PI baseline": "pressure_index",
    "PI RRR-heavy": "pi_rrr_situation_heavy",
    "PI momentum-heavy": "pi_momentum_heavy",
    "PI stakes-heavy": "pi_stakes_heavy",
    "XGBoost": "xgb_pressure_score",
}

CREDITED_BOWLER_DISMISSALS = {
    "caught", "bowled", "lbw", "stumped", "stump", "hit wicket",
}


def _summarise_role(rows: pd.DataFrame, role: str, prior_deliveries: int) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()

    if role == "Batter":
        d = rows.copy()
        d["player_id"] = pd.to_numeric(d["striker_id"], errors="coerce")
        d["player_name"] = d["striker_name"].fillna("Unknown")
        d["performance_count"] = pd.to_numeric(d["runs"], errors="coerce").fillna(0.0)
        d["dismissal_of_player"] = (
            pd.to_numeric(d.get("dismissed_player_id"), errors="coerce")
            == pd.to_numeric(d["striker_id"], errors="coerce")
        ).astype(int)
        grouped = d.groupby(["player_id", "player_name"], dropna=False).agg(
            pressure_deliveries=("match_id", "size"),
            pressure_matches=("match_id", "nunique"),
            batter_runs=("performance_count", "sum"),
            dismissals=("dismissal_of_player", "sum"),
        ).reset_index()
        metric_name = "runs_per_100_pressure_deliveries"
    else:
        d = rows.copy()
        d["player_id"] = pd.to_numeric(d["bowler_id"], errors="coerce")
        d["player_name"] = d["bowler_name"].fillna("Unknown")
        dismissal = d.get("dismissal_type", pd.Series("", index=d.index)).fillna("").astype(str).str.strip().str.lower()
        d["performance_count"] = (
            d["is_wicket"].fillna(False).astype(bool) & dismissal.isin(CREDITED_BOWLER_DISMISSALS)
        ).astype(int)
        grouped = d.groupby(["player_id", "player_name"], dropna=False).agg(
            pressure_deliveries=("match_id", "size"),
            pressure_matches=("match_id", "nunique"),
            credited_wickets=("performance_count", "sum"),
        ).reset_index()
        metric_name = "wickets_per_100_pressure_deliveries"

    denominator = grouped["pressure_deliveries"].clip(lower=1)
    count_col = "batter_runs" if role == "Batter" else "credited_wickets"
    total_deliveries = float(grouped["pressure_deliveries"].sum())
    population_rate = (
        100.0 * float(grouped[count_col].sum()) / total_deliveries
        if total_deliveries > 0 else 0.0
    )
    grouped["observed_per_100"] = 100.0 * grouped[count_col] / denominator
    prior_count = prior_deliveries * population_rate / 100.0
    grouped["shrunk_per_100"] = (
        100.0 * (grouped[count_col] + prior_count) /
        (grouped["pressure_deliveries"] + prior_deliveries)
    )
    grouped["performance_metric"] = metric_name
    grouped["role"] = role
    return grouped


def build_player_rankings(
    real: pd.DataFrame,
    min_deliveries: int = DEFAULT_MIN_DELIVERIES,
    min_matches: int = DEFAULT_MIN_MATCHES,
    prior_deliveries: int = DEFAULT_PRIOR_DELIVERIES,
    high_pressure_threshold: float = 60.0,
) -> pd.DataFrame:
    """Create comparable high-pressure player tables under each pressure model.

    For every scoring method, high-pressure deliveries are score >= threshold.
    Batter metric: batter runs per 100 balls faced.
    Bowler metric: credited wickets per 100 balls bowled.
    Rates are shrunk toward the overall delivery-weighted rate using a
    fixed pseudo-sample of prior_deliveries. Eligibility is applied separately.
    """
    if real.empty:
        return pd.DataFrame()

    missing = [c for c in ["match_id", "striker_id", "bowler_id", "pressure_index", "xgb_pressure_score"]
               if c not in real.columns]
    if missing:
        raise ValueError(f"Missing player-ranking columns: {missing}")

    frames = []
    for method, score_col in SCORE_METHODS.items():
        if score_col not in real.columns:
            continue
        pressure_rows = real.loc[pd.to_numeric(real[score_col], errors="coerce") >= high_pressure_threshold].copy()
        for role in ("Batter", "Bowler"):
            summary = _summarise_role(pressure_rows, role, prior_deliveries)
            if summary.empty:
                continue
            summary["scoring_method"] = method
            summary["score_column"] = score_col
            summary["high_pressure_threshold"] = high_pressure_threshold
            summary["eligible"] = (
                summary["pressure_deliveries"].ge(min_deliveries)
                & summary["pressure_matches"].ge(min_matches)
                & summary["player_id"].notna()
            )
            summary["min_deliveries"] = int(min_deliveries)
            summary["min_matches"] = int(min_matches)
            summary["prior_deliveries"] = int(prior_deliveries)
            summary["rank"] = np.nan
            eligible_idx = summary.index[summary["eligible"]]
            summary.loc[eligible_idx, "rank"] = (
                summary.loc[eligible_idx, "shrunk_per_100"]
                .rank(method="min", ascending=False)
            )
            frames.append(summary)

    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values(["role", "scoring_method", "eligible", "rank", "shrunk_per_100"],
                     ascending=[True, True, False, True, False], na_position="last")
        .reset_index(drop=True)
    )


def compare_top10_rankings(rankings: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Long-form Top-N table suitable for dashboards and CSV review."""
    if rankings.empty:
        return pd.DataFrame()
    return (
        rankings.loc[rankings["eligible"] & rankings["rank"].le(top_n)]
        .sort_values(["role", "scoring_method", "rank"])
        .copy()
    )
