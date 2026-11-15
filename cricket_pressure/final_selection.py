from __future__ import annotations

import numpy as np
import pandas as pd

from .player_rankings import RECENCY_HALF_LIVES_DAYS, SCORE_METHODS

# The synthetic-trained XGBoost score is retained for sensitivity analysis,
# but is deliberately excluded from the final PI consensus.
CONSENSUS_METHODS = tuple(name for name in SCORE_METHODS if name != "XGBoost")
HALF_LIFE_LABELS = ("unweighted", *(f"{days}d" for days in RECENCY_HALF_LIVES_DAYS))
DEFAULT_HIGH_PRESSURE_THRESHOLD = 60.0
DEFAULT_PRIOR_BALLS = 100


def _mode_or_unknown(values: pd.Series) -> str:
    values = values.dropna().astype(str)
    if values.empty:
        return "Unknown"
    modes = values.mode()
    return str(modes.iloc[0]) if not modes.empty else str(values.iloc[0])


def _role_observations(rows: pd.DataFrame, role: str) -> pd.DataFrame:
    """Attach player outcome and role-specific exposure columns to delivery rows."""
    d = rows.copy()
    if role == "Batter":
        d["player_id"] = pd.to_numeric(d["striker_id"], errors="coerce")
        d["player_name"] = d["striker_name"].fillna("Unknown")
        d["metric_numerator"] = pd.to_numeric(d["runs"], errors="coerce").fillna(0.0)
        # A wide is not a ball faced; a no-ball is a ball faced.
        d["metric_denominator"] = (~d.get(
            "is_wide", pd.Series(False, index=d.index)
        ).fillna(False).astype(bool)).astype(float)
        d["dismissal_of_player"] = (
            pd.to_numeric(d.get("dismissed_player_id"), errors="coerce")
            == d["player_id"]
        ).astype(float)
    else:
        d["player_id"] = pd.to_numeric(d["bowler_id"], errors="coerce")
        d["player_name"] = d["bowler_name"].fillna("Unknown")
        dismissal = d.get(
            "dismissal_type", pd.Series("", index=d.index)
        ).fillna("").astype(str).str.strip().str.lower()
        credited_types = {"caught", "bowled", "lbw", "stumped", "stump", "hit wicket"}
        d["metric_numerator"] = (
            d["is_wicket"].fillna(False).astype(bool) & dismissal.isin(credited_types)
        ).astype(float)
        d["metric_denominator"] = d.get(
            "is_legal_ball", pd.Series(True, index=d.index)
        ).fillna(False).astype(bool).astype(float)
        d["dismissal_of_player"] = 0.0

    return d.loc[d["player_id"].notna()].copy()


def _aggregate_pressure_and_baseline(
    rows: pd.DataFrame,
    role: str,
    score_col: str,
    half_life_label: str,
    high_pressure_threshold: float,
    min_deliveries: int,
    min_matches: int,
    prior_balls: int,
) -> pd.DataFrame:
    """Compare pressure and lower-pressure chase deliveries within format-season strata."""
    d = _role_observations(rows, role)
    if d.empty:
        return pd.DataFrame()

    d["pressure_state"] = pd.to_numeric(d[score_col], errors="coerce").ge(
        high_pressure_threshold
    )
    d = d.loc[pd.to_numeric(d[score_col], errors="coerce").notna()].copy()
    if d.empty:
        return pd.DataFrame()

    d["weighted_numerator"] = d["metric_numerator"] * d["recency_weight"]
    d["weighted_denominator"] = d["metric_denominator"] * d["recency_weight"]
    d["weighted_dismissals"] = d["dismissal_of_player"] * d["recency_weight"]
    stratum_keys = ["player_id", "match_format", "season"]

    def aggregate(mask: pd.Series, prefix: str) -> pd.DataFrame:
        part = d.loc[mask]
        if part.empty:
            return pd.DataFrame()
        result = part.groupby(stratum_keys, dropna=False).agg(
            **{
                f"{prefix}_deliveries": ("match_id", "size"),
                f"{prefix}_matches": ("match_id", "nunique"),
                f"{prefix}_raw_num": ("metric_numerator", "sum"),
                f"{prefix}_raw_den": ("metric_denominator", "sum"),
                f"{prefix}_weighted_num": ("weighted_numerator", "sum"),
                f"{prefix}_weighted_den": ("weighted_denominator", "sum"),
                f"{prefix}_weighted_dismissals": ("weighted_dismissals", "sum"),
            }
        ).reset_index()
        return result

    pressure = aggregate(d["pressure_state"], "pressure")
    normal = aggregate(~d["pressure_state"], "normal")
    if pressure.empty or normal.empty:
        return pd.DataFrame()

    # Match only the same player, format and season; do not compare IPL pressure
    # deliveries with an unrelated T20I or season baseline.
    matched = pressure.merge(
        normal, on=stratum_keys, how="inner", validate="one_to_one"
    )
    if matched.empty:
        return pd.DataFrame()

    pressure_population_rate = (
        100.0 * matched["pressure_weighted_num"].sum()
        / matched["pressure_weighted_den"].sum()
        if matched["pressure_weighted_den"].sum() > 0 else np.nan
    )
    normal_population_rate = (
        100.0 * matched["normal_weighted_num"].sum()
        / matched["normal_weighted_den"].sum()
        if matched["normal_weighted_den"].sum() > 0 else np.nan
    )

    sum_columns = [
        "pressure_deliveries", "pressure_matches", "pressure_raw_num", "pressure_raw_den",
        "pressure_weighted_num", "pressure_weighted_den", "pressure_weighted_dismissals",
        "normal_deliveries", "normal_matches", "normal_raw_num", "normal_raw_den",
        "normal_weighted_num", "normal_weighted_den", "normal_weighted_dismissals",
    ]
    player = matched.groupby("player_id", as_index=False)[sum_columns].sum()
    names = d.groupby("player_id")["player_name"].agg(_mode_or_unknown).rename("player_name")
    player = player.merge(names, on="player_id", how="left", validate="one_to_one")

    player["pressure_rate_raw"] = (
        100.0 * player["pressure_raw_num"] / player["pressure_raw_den"].replace(0, np.nan)
    )
    player["normal_rate_raw"] = (
        100.0 * player["normal_raw_num"] / player["normal_raw_den"].replace(0, np.nan)
    )

    # Empirical-Bayes-style rate stabilization: both rates receive a 100-ball
    # pseudo-sample from their own role-wide, pressure/control population rate.
    player["pressure_rate"] = (
        100.0 * (player["pressure_weighted_num"] + prior_balls * pressure_population_rate / 100.0)
        / (player["pressure_weighted_den"] + prior_balls)
    )
    player["normal_rate"] = (
        100.0 * (player["normal_weighted_num"] + prior_balls * normal_population_rate / 100.0)
        / (player["normal_weighted_den"] + prior_balls)
    )
    player["uplift_per_100"] = player["pressure_rate"] - player["normal_rate"]
    player["uplift_pct"] = np.where(
        player["normal_rate"].gt(0),
        100.0 * player["uplift_per_100"] / player["normal_rate"],
        np.nan,
    )

    if role == "Batter":
        player["pressure_dismissals_per_100"] = (
            100.0 * player["pressure_weighted_dismissals"]
            / player["pressure_weighted_den"].replace(0, np.nan)
        )
        player["normal_dismissals_per_100"] = (
            100.0 * player["normal_weighted_dismissals"]
            / player["normal_weighted_den"].replace(0, np.nan)
        )
        player["dismissal_rate_delta_per_100"] = (
            player["pressure_dismissals_per_100"] - player["normal_dismissals_per_100"]
        )
        metric = "Runs per 100 balls faced"
    else:
        player["pressure_dismissals_per_100"] = np.nan
        player["normal_dismissals_per_100"] = np.nan
        player["dismissal_rate_delta_per_100"] = np.nan
        metric = "Credited wickets per 100 legal balls"

    player["role"] = role
    player["scoring_method"] = next(
        name for name, col in SCORE_METHODS.items() if col == score_col
    )
    player["score_column"] = score_col
    player["half_life"] = half_life_label
    player["performance_metric"] = metric
    player["pressure_population_rate"] = pressure_population_rate
    player["normal_population_rate"] = normal_population_rate
    player["eligible"] = (
        player["pressure_deliveries"].ge(min_deliveries)
        & player["pressure_matches"].ge(min_matches)
        & player["normal_deliveries"].ge(min_deliveries)
        & player["normal_matches"].ge(min_matches)
        & player["pressure_raw_den"].gt(0)
        & player["normal_raw_den"].gt(0)
        & player["uplift_pct"].notna()
    )
    player["rank"] = np.nan
    eligible_idx = player.index[player["eligible"]]
    player.loc[eligible_idx, "rank"] = (
        player.loc[eligible_idx, "uplift_pct"].rank(method="min", ascending=False)
    )
    player["rank_percentile"] = np.nan
    player.loc[eligible_idx, "rank_percentile"] = (
        player.loc[eligible_idx, "uplift_pct"].rank(method="average", pct=True, ascending=True)
    )
    player["min_pressure_deliveries"] = int(min_deliveries)
    player["min_pressure_matches"] = int(min_matches)
    player["min_baseline_deliveries"] = int(min_deliveries)
    player["min_baseline_matches"] = int(min_matches)
    player["prior_balls"] = int(prior_balls)
    return player


def build_final_selection_candidates(
    real: pd.DataFrame,
    min_deliveries: int = 100,
    min_matches: int = 10,
    prior_balls: int = DEFAULT_PRIOR_BALLS,
    high_pressure_threshold: float = DEFAULT_HIGH_PRESSURE_THRESHOLD,
    half_lives_days: tuple[int, ...] = RECENCY_HALF_LIVES_DAYS,
) -> pd.DataFrame:
    """Build pressure-uplift candidates for all score methods and recency settings.

    The comparison baseline is the same player's sub-threshold *chase* deliveries
    within the same format and season. It is not the player's all-innings career
    average. Eligibility applies the sample gates to both pressure and baseline
    exposures so that a thin control sample cannot qualify.
    """
    if real.empty:
        return pd.DataFrame()

    required = {
        "match_id", "match_date", "match_format", "season",
        "striker_id", "striker_name", "bowler_id", "bowler_name",
        "runs", "is_wicket", "dismissed_player_id", "dismissal_type",
        "is_wide", "is_legal_ball",
    }
    missing = sorted(required - set(real.columns))
    if missing:
        raise ValueError(f"Missing final-selection columns: {missing}")

    parsed_dates = pd.to_datetime(real["match_date"], errors="coerce", utc=True)
    if parsed_dates.notna().sum() == 0:
        raise ValueError("Final selection requires valid match_date values")
    reference_date = parsed_dates.max()
    frames: list[pd.DataFrame] = []

    for method, score_col in SCORE_METHODS.items():
        if score_col not in real.columns:
            continue
        base = real.copy()
        base["match_date"] = parsed_dates
        for half_life in (None, *half_lives_days):
            if half_life is None:
                label = "unweighted"
                base["recency_weight"] = 1.0
            else:
                label = f"{half_life}d"
                age_days = (reference_date - base["match_date"]).dt.total_seconds() / 86400.0
                base["recency_weight"] = np.power(
                    2.0, -age_days.clip(lower=0) / float(half_life)
                )
                base.loc[base["match_date"].isna(), "recency_weight"] = np.nan
                base = base.loc[base["recency_weight"].notna()].copy()

            for role in ("Batter", "Bowler"):
                summary = _aggregate_pressure_and_baseline(
                    base, role, score_col, label, high_pressure_threshold,
                    min_deliveries, min_matches, prior_balls,
                )
                if not summary.empty:
                    summary["recency_reference_date"] = reference_date.isoformat()
                    frames.append(summary)

    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values(
            ["role", "scoring_method", "half_life", "eligible", "rank", "uplift_pct"],
            ascending=[True, True, True, False, True, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def build_consensus_rankings(
    candidates: pd.DataFrame,
    min_methods: int = 3,
) -> pd.DataFrame:
    """Rank players by median PI-method percentile, excluding synthetic XGBoost."""
    if candidates.empty:
        return pd.DataFrame()
    pi_rows = candidates.loc[
        candidates["scoring_method"].isin(CONSENSUS_METHODS) & candidates["eligible"]
    ].copy()
    if pi_rows.empty:
        return pd.DataFrame()

    consensus = pi_rows.groupby(
        ["role", "half_life", "player_id"], as_index=False
    ).agg(
        consensus_percentile=("rank_percentile", "median"),
        methods_supported=("scoring_method", "nunique"),
        player_name=("player_name", _mode_or_unknown),
    )
    consensus = consensus.loc[consensus["methods_supported"].ge(min_methods)].copy()
    if consensus.empty:
        return pd.DataFrame()

    consensus["consensus_rank"] = consensus.groupby(
        ["role", "half_life"]
    )["consensus_percentile"].rank(method="min", ascending=False)

    # Use PI baseline metrics for a consistent, interpretable explanation column.
    baseline = candidates.loc[
        candidates["scoring_method"].eq("PI baseline"),
        [
            "role", "half_life", "player_id", "pressure_rate", "normal_rate",
            "pressure_rate_raw", "normal_rate_raw", "uplift_per_100", "uplift_pct",
            "pressure_deliveries", "pressure_matches", "normal_deliveries",
            "normal_matches", "pressure_dismissals_per_100",
            "normal_dismissals_per_100", "dismissal_rate_delta_per_100",
            "performance_metric", "eligible",
        ],
    ].rename(columns={
        "pressure_rate": "baseline_pi_pressure_rate",
        "normal_rate": "baseline_pi_normal_rate",
        "pressure_rate_raw": "baseline_pi_pressure_rate_raw",
        "normal_rate_raw": "baseline_pi_normal_rate_raw",
        "uplift_per_100": "baseline_pi_uplift_per_100",
        "uplift_pct": "baseline_pi_uplift_pct",
        "pressure_deliveries": "baseline_pi_pressure_deliveries",
        "pressure_matches": "baseline_pi_pressure_matches",
        "normal_deliveries": "baseline_pi_normal_deliveries",
        "normal_matches": "baseline_pi_normal_matches",
        "pressure_dismissals_per_100": "baseline_pi_pressure_dismissals_per_100",
        "normal_dismissals_per_100": "baseline_pi_normal_dismissals_per_100",
        "dismissal_rate_delta_per_100": "baseline_pi_dismissal_rate_delta_per_100",
        "performance_metric": "performance_metric",
        "eligible": "baseline_pi_eligible",
    })
    consensus = consensus.merge(
        baseline, on=["role", "half_life", "player_id"], how="left", validate="one_to_one"
    )
    consensus["explanation"] = consensus.apply(
        lambda row: (
            f"{row['baseline_pi_uplift_pct']:+.1f}% vs lower-pressure chase baseline "
            f"({row['baseline_pi_normal_rate']:.1f} → {row['baseline_pi_pressure_rate']:.1f} "
            f"per 100); {int(row['baseline_pi_pressure_deliveries']):,} pressure deliveries "
            f"across {int(row['baseline_pi_pressure_matches']):,} matches; "
            f"supported by {int(row['methods_supported'])}/{len(CONSENSUS_METHODS)} PI variants."
        )
        if pd.notna(row.get("baseline_pi_uplift_pct"))
        and pd.notna(row.get("baseline_pi_pressure_deliveries"))
        else (
            f"Median PI-method percentile: {row['consensus_percentile']:.3f}; "
            f"supported by {int(row['methods_supported'])}/{len(CONSENSUS_METHODS)} PI variants. "
            "Baseline PI summary unavailable."
        ),
        axis=1,
    )
    return consensus.sort_values(
        ["role", "half_life", "consensus_rank", "consensus_percentile"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
