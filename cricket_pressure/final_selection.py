from __future__ import annotations

import numpy as np
import pandas as pd

from .player_rankings import RECENCY_HALF_LIVES_DAYS, SCORE_METHODS

# The synthetic-trained XGBoost score is a diagnostic only. Consensus is based
# on the four transparent PI definitions, not synthetic training labels.
CONSENSUS_METHODS = tuple(name for name in SCORE_METHODS if name != "XGBoost")
HALF_LIFE_LABELS = ("unweighted", *(f"{days}d" for days in RECENCY_HALF_LIVES_DAYS))
DEFAULT_HIGH_PRESSURE_THRESHOLD = 60.0
DEFAULT_PRIOR_MATCHES = 5
DEFAULT_RISK_PENALTY = 1.0


def _mode_or_unknown(values: pd.Series) -> str:
    values = values.dropna().astype(str)
    if values.empty:
        return "Unknown"
    modes = values.mode()
    return str(modes.iloc[0]) if not modes.empty else str(values.iloc[0])


def _role_observations(rows: pd.DataFrame, role: str) -> pd.DataFrame:
    """Attach role-specific, pressure-performance outcomes to delivery rows."""
    d = rows.copy()
    if role == "Batter":
        d["player_id"] = pd.to_numeric(d["striker_id"], errors="coerce")
        d["player_name"] = d["striker_name"].fillna("Unknown")
        d["performance_numerator"] = pd.to_numeric(
            d["runs"], errors="coerce"
        ).fillna(0.0)
        # A wide is not a ball faced; a no-ball is a ball faced.
        d["performance_denominator"] = (
            ~d["is_wide"].fillna(False).astype(bool)
        ).astype(float)
        d["secondary_numerator"] = (
            pd.to_numeric(d["dismissed_player_id"], errors="coerce")
            == d["player_id"]
        ).astype(float)
        d["secondary_denominator"] = d["performance_denominator"]
    else:
        d["player_id"] = pd.to_numeric(d["bowler_id"], errors="coerce")
        d["player_name"] = d["bowler_name"].fillna("Unknown")
        legal = d["is_legal_ball"].fillna(False).astype(bool)
        unresolved = d["unresolved_bye_legbye"].fillna(False).astype(bool)
        known_legal = legal & ~unresolved
        total_runs = pd.to_numeric(d["provisional_total_runs"], errors="coerce")
        # Dot-ball rate is the stable primary bowling measure. Unknown bye /
        # legbye rows are excluded rather than incorrectly treated as dots.
        d["performance_numerator"] = (
            known_legal & total_runs.eq(0)
        ).astype(float)
        d["performance_denominator"] = known_legal.astype(float)
        dismissal = d["dismissal_type"].fillna("").astype(str).str.strip().str.lower()
        credited_types = {"caught", "bowled", "lbw", "stumped", "stump", "hit wicket"}
        d["secondary_numerator"] = (
            d["is_wicket"].fillna(False).astype(bool) & dismissal.isin(credited_types)
        ).astype(float)
        d["secondary_denominator"] = legal.astype(float)

    return d.loc[d["player_id"].notna()].copy()


def _weighted_reliability(
    match_rows: pd.DataFrame,
    population_rate: float,
    prior_matches: int,
    risk_penalty: float,
) -> dict[str, float]:
    """Return a shrunk mean minus a standard-error penalty over match rates."""
    valid = match_rows.loc[
        match_rows["match_rate"].notna()
        & match_rows["performance_denominator"].gt(0)
        & match_rows["recency_weight"].notna()
    ].copy()
    if valid.empty:
        return {
            "mean_match_rate": np.nan, "std_match_rate": np.nan,
            "effective_matches": 0.0, "shrunk_mean_rate": np.nan,
            "consistency_score": np.nan, "consistency_pct": np.nan,
        }

    # Square-root exposure weighting keeps a long spell from dominating the
    # estimate while still down-weighting a one-ball match.
    exposure = np.sqrt(valid["performance_denominator"].clip(lower=1.0))
    weights = exposure * valid["recency_weight"].astype(float)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return {
            "mean_match_rate": np.nan, "std_match_rate": np.nan,
            "effective_matches": 0.0, "shrunk_mean_rate": np.nan,
            "consistency_score": np.nan, "consistency_pct": np.nan,
        }

    rates = valid["match_rate"].astype(float)
    mean_rate = float(np.average(rates, weights=weights))
    variance = float(np.average((rates - mean_rate) ** 2, weights=weights))
    std_rate = float(np.sqrt(max(variance, 0.0)))
    effective_matches = float(total_weight ** 2 / np.square(weights).sum())
    prior = max(int(prior_matches), 0)
    shrunk_mean = (
        (mean_rate * effective_matches + population_rate * prior)
        / (effective_matches + prior)
        if effective_matches + prior > 0 else mean_rate
    )
    # A one-standard-error lower bound rewards a strong mean that repeats
    # across matches and penalizes volatility / thin effective samples.
    standard_error = std_rate / np.sqrt(max(effective_matches + prior, 1.0))
    consistency_score = shrunk_mean - float(risk_penalty) * standard_error
    consistency_pct = float(np.average((rates >= population_rate).astype(float), weights=weights))
    return {
        "mean_match_rate": mean_rate,
        "std_match_rate": std_rate,
        "effective_matches": effective_matches,
        "shrunk_mean_rate": shrunk_mean,
        "consistency_score": consistency_score,
        "consistency_pct": consistency_pct,
    }


def _aggregate_player_matches(
    rows: pd.DataFrame,
    role: str,
    score_col: str,
    method: str,
    half_life_label: str,
    high_pressure_threshold: float,
    min_deliveries: int,
    min_matches: int,
    prior_matches: int,
    risk_penalty: float,
) -> pd.DataFrame:
    d = _role_observations(rows, role)
    if d.empty:
        return pd.DataFrame()

    scores = pd.to_numeric(d[score_col], errors="coerce")
    d = d.loc[scores.ge(high_pressure_threshold)].copy()
    if d.empty:
        return pd.DataFrame()

    # A match is the independent unit for consistency: aggregate pressure
    # outcomes within player-match before assessing mean and volatility.
    keys = ["player_id", "match_id"]
    match_rows = d.groupby(keys, dropna=False).agg(
        player_name=("player_name", _mode_or_unknown),
        match_date=("match_date", "max"),
        pressure_deliveries=("match_id", "size"),
        performance_numerator=("performance_numerator", "sum"),
        performance_denominator=("performance_denominator", "sum"),
        secondary_numerator=("secondary_numerator", "sum"),
        secondary_denominator=("secondary_denominator", "sum"),
        recency_weight=("recency_weight", "max"),
    ).reset_index()
    match_rows["match_rate"] = (
        100.0 * match_rows["performance_numerator"]
        / match_rows["performance_denominator"].replace(0, np.nan)
    )

    population_denominator = float(
        (match_rows["performance_denominator"] * match_rows["recency_weight"]).sum()
    )
    population_numerator = float(
        (match_rows["performance_numerator"] * match_rows["recency_weight"]).sum()
    )
    population_rate = (
        100.0 * population_numerator / population_denominator
        if population_denominator > 0 else np.nan
    )
    if not np.isfinite(population_rate):
        return pd.DataFrame()

    output = []
    for player_id, player_matches in match_rows.groupby("player_id", sort=False):
        player_matches = player_matches.copy()
        reliability = _weighted_reliability(
            player_matches, population_rate, prior_matches, risk_penalty
        )
        valid_match_rows = player_matches.loc[player_matches["match_rate"].notna()].copy()
        pressure_deliveries = int(valid_match_rows["performance_denominator"].sum())
        pressure_matches = int(valid_match_rows["match_id"].nunique())
        raw_denominator = float(player_matches["performance_denominator"].sum())
        raw_numerator = float(player_matches["performance_numerator"].sum())
        weighted_denominator = float(
            (player_matches["performance_denominator"] * player_matches["recency_weight"]).sum()
        )
        weighted_numerator = float(
            (player_matches["performance_numerator"] * player_matches["recency_weight"]).sum()
        )
        secondary_denominator = float(
            (player_matches["secondary_denominator"] * player_matches["recency_weight"]).sum()
        )
        secondary_numerator = float(
            (player_matches["secondary_numerator"] * player_matches["recency_weight"]).sum()
        )
        observed_rate = (
            100.0 * raw_numerator / raw_denominator if raw_denominator > 0 else np.nan
        )
        weighted_rate = (
            100.0 * weighted_numerator / weighted_denominator
            if weighted_denominator > 0 else np.nan
        )
        secondary_rate = (
            100.0 * secondary_numerator / secondary_denominator
            if secondary_denominator > 0 else np.nan
        )
        eligible = (
            pressure_deliveries >= int(min_deliveries)
            and pressure_matches >= int(min_matches)
            and np.isfinite(reliability["consistency_score"])
        )
        output.append({
            "player_id": player_id,
            "player_name": _mode_or_unknown(player_matches["player_name"]),
            "role": role,
            "scoring_method": method,
            "score_column": score_col,
            "half_life": half_life_label,
            "pressure_deliveries": pressure_deliveries,
            "pressure_matches": pressure_matches,
            "raw_pressure_rate": observed_rate,
            "recency_weighted_pressure_rate": weighted_rate,
            "population_pressure_rate": population_rate,
            "secondary_rate": secondary_rate,
            "mean_match_rate": reliability["mean_match_rate"],
            "std_match_rate": reliability["std_match_rate"],
            "effective_matches": reliability["effective_matches"],
            "shrunk_mean_rate": reliability["shrunk_mean_rate"],
            "consistency_score": reliability["consistency_score"],
            "consistency_pct": reliability["consistency_pct"],
            "eligible": eligible,
            "min_deliveries": int(min_deliveries),
            "min_matches": int(min_matches),
            "prior_matches": int(prior_matches),
            "risk_penalty": float(risk_penalty),
            "performance_metric": (
                "Runs per 100 balls faced" if role == "Batter"
                else "Dot-ball percentage of known legal balls"
            ),
            "secondary_metric": (
                "Dismissals per 100 balls faced" if role == "Batter"
                else "Credited wickets per 100 legal balls"
            ),
        })

    result = pd.DataFrame(output)
    result["rank"] = np.nan
    eligible_idx = result.index[result["eligible"]]
    result.loc[eligible_idx, "rank"] = (
        result.loc[eligible_idx, "consistency_score"].rank(method="min", ascending=False)
    )
    result["rank_percentile"] = np.nan
    result.loc[eligible_idx, "rank_percentile"] = (
        result.loc[eligible_idx, "consistency_score"]
        .rank(method="average", pct=True, ascending=True)
    )
    return result


def build_final_selection_candidates(
    real: pd.DataFrame,
    min_deliveries: int = 100,
    min_matches: int = 10,
    prior_matches: int = DEFAULT_PRIOR_MATCHES,
    risk_penalty: float = DEFAULT_RISK_PENALTY,
    high_pressure_threshold: float = DEFAULT_HIGH_PRESSURE_THRESHOLD,
    half_lives_days: tuple[int, ...] = RECENCY_HALF_LIVES_DAYS,
) -> pd.DataFrame:
    """Build match-consistency pressure rankings, not pressure-vs-baseline uplift.

    Batters are assessed by runs per 100 balls faced in high-pressure chase
    deliveries. Bowlers use dot-ball percentage on known legal deliveries;
    unresolved bye/legbye rows are excluded from that denominator. The ranking
    score is a shrunk mean match rate minus a standard-error penalty, so strong
    repeatable performances outrank volatile rates from thin effective samples.
    """
    if real.empty:
        return pd.DataFrame()

    required = {
        "match_id", "match_date", "striker_id", "striker_name", "bowler_id",
        "bowler_name", "runs", "is_wide", "is_legal_ball",
        "unresolved_bye_legbye", "provisional_total_runs", "is_wicket",
        "dismissed_player_id", "dismissal_type",
    }
    missing = sorted(required - set(real.columns))
    if missing:
        raise ValueError(f"Missing final-selection columns: {missing}")

    parsed_dates = pd.to_datetime(real["match_date"], errors="coerce", utc=True)
    valid_dates = parsed_dates.notna()
    if not valid_dates.any():
        raise ValueError("Final selection requires valid match_date values")
    reference_date = parsed_dates.max()

    base = real.copy()
    base["match_date"] = parsed_dates
    base = base.loc[valid_dates].copy()
    frames: list[pd.DataFrame] = []

    for method, score_col in SCORE_METHODS.items():
        if score_col not in base.columns:
            continue
        for half_life in (None, *half_lives_days):
            if half_life is None:
                label = "unweighted"
                sample = base.copy()
                sample["recency_weight"] = 1.0
            else:
                label = f"{half_life}d"
                sample = base.copy()
                age_days = (
                    reference_date - sample["match_date"]
                ).dt.total_seconds() / 86400.0
                sample["recency_weight"] = np.power(
                    2.0, -age_days.clip(lower=0) / float(half_life)
                )
            for role in ("Batter", "Bowler"):
                summary = _aggregate_player_matches(
                    sample, role, score_col, method, label,
                    high_pressure_threshold, min_deliveries, min_matches,
                    prior_matches, risk_penalty,
                )
                if not summary.empty:
                    summary["recency_reference_date"] = reference_date.isoformat()
                    summary["high_pressure_threshold"] = float(high_pressure_threshold)
                    frames.append(summary)

    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values(
            ["role", "scoring_method", "half_life", "eligible", "rank", "consistency_score"],
            ascending=[True, True, True, False, True, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def build_consensus_rankings(
    candidates: pd.DataFrame,
    min_methods: int = 3,
) -> pd.DataFrame:
    """Rank candidates by median percentile across the four PI definitions."""
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

    baseline = candidates.loc[
        candidates["scoring_method"].eq("PI baseline"),
        [
            "role", "half_life", "player_id", "raw_pressure_rate",
            "recency_weighted_pressure_rate", "population_pressure_rate",
            "secondary_rate", "mean_match_rate", "std_match_rate",
            "effective_matches", "shrunk_mean_rate", "consistency_score",
            "consistency_pct", "pressure_deliveries", "pressure_matches",
            "performance_metric", "secondary_metric", "eligible",
        ],
    ].rename(columns={
        "raw_pressure_rate": "baseline_pi_raw_pressure_rate",
        "recency_weighted_pressure_rate": "baseline_pi_weighted_pressure_rate",
        "population_pressure_rate": "baseline_pi_population_rate",
        "secondary_rate": "baseline_pi_secondary_rate",
        "mean_match_rate": "baseline_pi_mean_match_rate",
        "std_match_rate": "baseline_pi_std_match_rate",
        "effective_matches": "baseline_pi_effective_matches",
        "shrunk_mean_rate": "baseline_pi_shrunk_mean_rate",
        "consistency_score": "baseline_pi_consistency_score",
        "consistency_pct": "baseline_pi_consistency_pct",
        "pressure_deliveries": "baseline_pi_pressure_deliveries",
        "pressure_matches": "baseline_pi_pressure_matches",
        "performance_metric": "performance_metric",
        "secondary_metric": "secondary_metric",
        "eligible": "baseline_pi_eligible",
    })
    consensus = consensus.merge(
        baseline, on=["role", "half_life", "player_id"], how="left", validate="one_to_one"
    )
    consensus["explanation"] = consensus.apply(
        lambda row: (
            f"Consistency score {row['baseline_pi_consistency_score']:.2f} "
            f"(PI baseline); average pressure-match rate "
            f"{row['baseline_pi_mean_match_rate']:.2f} with match SD "
            f"{row['baseline_pi_std_match_rate']:.2f}; "
            f"{int(row['baseline_pi_pressure_deliveries']):,} pressure deliveries "
            f"across {int(row['baseline_pi_pressure_matches']):,} matches; "
            f"supported by {int(row['methods_supported'])}/{len(CONSENSUS_METHODS)} PI variants."
        )
        if pd.notna(row.get("baseline_pi_consistency_score"))
        and pd.notna(row.get("baseline_pi_pressure_deliveries"))
        else (
            f"Median PI-method percentile: {row['consensus_percentile']:.3f}; "
            f"supported by {int(row['methods_supported'])}/{len(CONSENSUS_METHODS)} PI variants."
        ),
        axis=1,
    )
    return consensus.sort_values(
        ["role", "half_life", "consensus_rank", "consensus_percentile"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
