from __future__ import annotations

import numpy as np
import pandas as pd

# These are starting research settings, not universal sample-size truths.
DEFAULT_MIN_DELIVERIES = 100
DEFAULT_MIN_MATCHES = 10
DEFAULT_PRIOR_DELIVERIES = 100
RECENCY_HALF_LIVES_DAYS = (365, 540, 730)

SCORE_METHODS = {
    "PI baseline": "pressure_index",
    "PI RRR-heavy": "pi_rrr_situation_heavy",
    "PI momentum-heavy": "pi_momentum_heavy",
    "PI stakes-heavy": "pi_stakes_heavy",
    "XGBoost": "xgb_pressure_score",
}


def exponential_recency_weights(
    dates: pd.Series,
    half_life_days: int | float,
    reference_date: pd.Timestamp | str | None = None,
) -> tuple[pd.Series, pd.Timestamp]:
    """Return 2**(-age_days / half_life), anchored to a fixed reference date.

    By default the reference date is the latest valid date in the supplied
    data, making results reproducible when rerunning the same dataset.
    """
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    parsed = pd.to_datetime(dates, errors="coerce", utc=True)
    if reference_date is None:
        if parsed.notna().sum() == 0:
            raise ValueError("Cannot calculate recency weights: no valid dates")
        reference = parsed.max()
    else:
        reference = pd.to_datetime(reference_date, errors="raise", utc=True)
    age_days = (reference - parsed).dt.total_seconds() / 86400.0
    # Dates after the reference are treated as age zero and explicitly flagged upstream.
    age_days = age_days.clip(lower=0)
    weights = pd.Series(
        np.power(2.0, -age_days / float(half_life_days)),
        index=dates.index,
        name="recency_weight",
    )
    weights = weights.where(parsed.notna(), np.nan)
    return weights, pd.Timestamp(reference)


def _summarise_role(
    rows: pd.DataFrame,
    role: str,
    prior_deliveries: int,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()

    d = rows.copy()
    if role == "Batter":
        d["player_id"] = pd.to_numeric(d["striker_id"], errors="coerce")
        d["player_name"] = d["striker_name"].fillna("Unknown")
        d["performance_count"] = pd.to_numeric(d["runs"], errors="coerce").fillna(0.0)
        d["dismissal_of_player"] = (
            pd.to_numeric(d.get("dismissed_player_id"), errors="coerce")
            == pd.to_numeric(d["striker_id"], errors="coerce")
        ).astype(int)
        d["credited_wicket"] = 0
        count_col = "performance_count"
        metric_name = "runs_per_100_pressure_deliveries"
    else:
        d["player_id"] = pd.to_numeric(d["bowler_id"], errors="coerce")
        d["player_name"] = d["bowler_name"].fillna("Unknown")
        dismissal = d.get("dismissal_type", pd.Series("", index=d.index)).fillna("").astype(str).str.strip().str.lower()
        credited_types = {"caught", "bowled", "lbw", "stumped", "stump", "hit wicket"}
        d["credited_wicket"] = (
            d["is_wicket"].fillna(False).astype(bool) & dismissal.isin(credited_types)
        ).astype(int)
        d["dismissal_of_player"] = 0
        count_col = "credited_wicket"
        metric_name = "wickets_per_100_pressure_deliveries"

    # recency_weight is one for unweighted ranking and exponential for decayed ranking.
    d["weighted_deliveries"] = d["recency_weight"].astype(float)
    d["weighted_performance"] = d[count_col].astype(float) * d["recency_weight"].astype(float)

    aggregations = {
        "pressure_deliveries": ("match_id", "size"),
        "pressure_matches": ("match_id", "nunique"),
        "raw_performance": (count_col, "sum"),
        "weighted_deliveries": ("weighted_deliveries", "sum"),
        "weighted_performance": ("weighted_performance", "sum"),
        "dismissals": ("dismissal_of_player", "sum"),
    }
    grouped = (
        d.groupby(["player_id", "player_name"], dropna=False)
        .agg(**aggregations)
        .reset_index()
    )

    # Weighted population baseline is computed across all eligible-role rows
    # within this pressure definition and half-life.
    total_weighted_deliveries = float(grouped["weighted_deliveries"].sum())
    population_rate = (
        100.0 * float(grouped["weighted_performance"].sum()) / total_weighted_deliveries
        if total_weighted_deliveries > 0 else 0.0
    )
    grouped["observed_per_100"] = (
        100.0 * grouped["raw_performance"] /
        grouped["pressure_deliveries"].replace(0, np.nan)
    )
    grouped["weighted_per_100"] = (
        100.0 * grouped["weighted_performance"] /
        grouped["weighted_deliveries"].replace(0, np.nan)
    )
    # Add a fixed prior of 100 weighted delivery-equivalents to stabilize thin samples.
    prior_performance = prior_deliveries * population_rate / 100.0
    grouped["shrunk_per_100"] = (
        100.0 * (grouped["weighted_performance"] + prior_performance) /
        (grouped["weighted_deliveries"] + prior_deliveries)
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
    half_lives_days: tuple[int, ...] = RECENCY_HALF_LIVES_DAYS,
) -> pd.DataFrame:
    """Build unweighted and recency-weighted high-pressure player rankings.

    Eligibility always uses raw counts: at least min_deliveries pressure balls
    across min_matches distinct matches. Recency changes the performance rate,
    not the sample gate. Shrinkage uses a fixed pseudo-sample prior of
    prior_deliveries weighted delivery-equivalents.

    The reference date is the newest valid match date in the supplied dataset,
    not the machine's current date. This makes rankings reproducible.
    """
    if real.empty:
        return pd.DataFrame()
    required = [
        "match_id", "match_date", "striker_id", "striker_name", "bowler_id",
        "bowler_name", "pressure_index", "xgb_pressure_score", "runs",
    ]
    missing = [c for c in required if c not in real.columns]
    if missing:
        raise ValueError(f"Missing player-ranking columns: {missing}")

    parsed_dates = pd.to_datetime(real["match_date"], errors="coerce", utc=True)
    if parsed_dates.notna().sum() == 0:
        raise ValueError("Player recency weighting requires valid match_date values")
    reference_date = parsed_dates.max()
    date_reference = pd.Timestamp(reference_date).isoformat()

    frames = []
    for method, score_col in SCORE_METHODS.items():
        if score_col not in real.columns:
            continue
        pressure_rows = real.loc[
            pd.to_numeric(real[score_col], errors="coerce") >= high_pressure_threshold
        ].copy()
        if pressure_rows.empty:
            continue
        pressure_rows["match_date"] = pd.to_datetime(
            pressure_rows["match_date"], errors="coerce", utc=True
        )

        recency_specs: list[tuple[str, int | None]] = [("unweighted", None)]
        recency_specs.extend((f"{h}d", h) for h in half_lives_days)
        for half_life_label, half_life_days in recency_specs:
            part = pressure_rows.copy()
            if half_life_days is None:
                part["recency_weight"] = 1.0
            else:
                age_days = (
                    reference_date - part["match_date"]
                ).dt.total_seconds() / 86400.0
                part["recency_weight"] = np.power(
                    2.0, -age_days.clip(lower=0) / float(half_life_days)
                )
                part.loc[part["match_date"].isna(), "recency_weight"] = np.nan
                part = part.loc[part["recency_weight"].notna()].copy()

            for role in ("Batter", "Bowler"):
                summary = _summarise_role(part, role, prior_deliveries)
                if summary.empty:
                    continue
                summary["scoring_method"] = method
                summary["score_column"] = score_col
                summary["half_life"] = half_life_label
                summary["half_life_days"] = half_life_days if half_life_days is not None else 0
                summary["recency_reference_date"] = date_reference
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
        .sort_values(
            ["role", "scoring_method", "half_life_days", "eligible", "rank", "shrunk_per_100"],
            ascending=[True, True, True, False, True, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def compare_top10_rankings(rankings: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Long-form Top-N comparison table for the dashboard and CSV exports."""
    if rankings.empty:
        return pd.DataFrame()
    return (
        rankings.loc[rankings["eligible"] & rankings["rank"].le(top_n)]
        .sort_values(["role", "half_life_days", "scoring_method", "rank"])
        .copy()
    )
