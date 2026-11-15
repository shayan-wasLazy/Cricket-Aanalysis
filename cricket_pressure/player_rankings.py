from __future__ import annotations

import numpy as np
import pandas as pd

# These are starting research settings, not universal sample-size truths.
DEFAULT_MIN_DELIVERIES = 100
DEFAULT_MIN_MATCHES = 10
DEFAULT_PRIOR_DELIVERIES = 100
# 180 days gives recent form substantially more influence while retaining
# longer half-lives for comparison.
RECENCY_HALF_LIVES_DAYS = (180, 365, 540, 730)
DEFAULT_PLAYER_RECENCY_HALF_LIFE = 180

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
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()

    d = rows.copy()
    if role == "Batter":
        # Wides do not count as balls faced; no-balls do.
        if "is_wide" in d.columns:
            d = d.loc[~d["is_wide"].fillna(False).astype(bool)].copy()
        if d.empty:
            return pd.DataFrame()
        d["player_id"] = pd.to_numeric(d["striker_id"], errors="coerce")
        d["player_name"] = d["striker_name"].fillna("Unknown")
        d["performance_count"] = pd.to_numeric(d["runs"], errors="coerce").fillna(0.0)
        d["dismissal_of_player"] = (
            pd.to_numeric(d.get("dismissed_player_id"), errors="coerce")
            == pd.to_numeric(d["striker_id"], errors="coerce")
        ).astype(int)
        d["credited_wicket"] = 0
        d["delivery_unit"] = 1.0
        d["opportunity_match_id"] = d["match_id"]
        count_col = "performance_count"
        metric_name = "runs_per_100_pressure_deliveries"
    else:
        # Keep all charged deliveries in the economy numerator. Only legal balls
        # count in the denominator, wicket rate and dot-ball rate.
        d["player_id"] = pd.to_numeric(d["bowler_id"], errors="coerce")
        d["player_name"] = d["bowler_name"].fillna("Unknown")
        d["is_legal_delivery"] = d.get(
            "is_legal_ball", pd.Series(True, index=d.index)
        ).fillna(False).astype(bool)
        dismissal = d.get("dismissal_type", pd.Series("", index=d.index)).fillna("").astype(str).str.strip().str.lower()
        credited_types = {"caught", "caught and bowled", "bowled", "lbw", "stumped", "stump", "hit wicket"}
        d["credited_wicket"] = (
            d["is_wicket"].fillna(False).astype(bool) & dismissal.isin(credited_types)
        ).astype(int)
        d["dismissal_of_player"] = 0
        extras_type = d.get("extras_type", pd.Series("", index=d.index)).fillna("").astype(str).str.lower().str.strip()
        extras_runs = pd.to_numeric(d.get("extras_2_runs", pd.Series(0, index=d.index)), errors="coerce").fillna(0)
        charged_extra = extras_runs.where(extras_type.isin({"wide", "overstep"}), 0)
        charged_extra = charged_extra + extras_type.isin({"wide", "overstep"}).astype(float)
        d["bowler_runs_conceded"] = (
            pd.to_numeric(d["runs"], errors="coerce").fillna(0) + charged_extra
        )
        # A bye/leg-bye is not charged to the bowler, but it is still not a dot ball.
        delivery_runs = pd.to_numeric(
            d.get("provisional_total_runs", d["bowler_runs_conceded"]),
            errors="coerce",
        ).fillna(0)
        d["bowler_dot"] = (
            delivery_runs.eq(0) & d["is_legal_delivery"]
        ).astype(int)
        d["delivery_unit"] = d["is_legal_delivery"].astype(float)
        d["opportunity_match_id"] = d["match_id"].where(d["is_legal_delivery"])
        d["weighted_runs_conceded"] = d["bowler_runs_conceded"] * d["recency_weight"].astype(float)
        d["weighted_dots"] = d["bowler_dot"] * d["recency_weight"].astype(float)
        count_col = "credited_wicket"
        metric_name = "wickets_per_100_pressure_deliveries"

    # Recency weights apply to the role's opportunity unit: batter balls faced
    # or bowler legal balls.
    d["weighted_deliveries"] = (
        d["recency_weight"].astype(float) * d["delivery_unit"].astype(float)
    )
    d["weighted_performance"] = d[count_col].astype(float) * d["recency_weight"].astype(float)

    aggregations = {
        "pressure_deliveries": ("delivery_unit", "sum"),
        "pressure_matches": ("opportunity_match_id", "nunique"),
        "raw_performance": (count_col, "sum"),
        "weighted_deliveries": ("weighted_deliveries", "sum"),
        "weighted_performance": ("weighted_performance", "sum"),
        "dismissals": ("dismissal_of_player", "sum"),
    }
    if role == "Bowler":
        aggregations.update({
            "raw_runs_conceded": ("bowler_runs_conceded", "sum"),
            "raw_dots": ("bowler_dot", "sum"),
            "weighted_runs_conceded": ("weighted_runs_conceded", "sum"),
            "weighted_dots": ("weighted_dots", "sum"),
        })
    grouped = (
        d.groupby(["player_id", "player_name"], dropna=False)
        .agg(**aggregations)
        .reset_index()
    )

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
    prior_performance = prior_deliveries * population_rate / 100.0
    grouped["shrunk_per_100"] = (
        100.0 * (grouped["weighted_performance"] + prior_performance) /
        (grouped["weighted_deliveries"] + prior_deliveries)
    )
    if role == "Bowler":
        grouped["economy_per_over"] = (
            6.0 * grouped["weighted_runs_conceded"] /
            grouped["weighted_deliveries"].replace(0, np.nan)
        )
        grouped["dot_rate"] = (
            100.0 * grouped["weighted_dots"] /
            grouped["weighted_deliveries"].replace(0, np.nan)
        )
    grouped["performance_metric"] = metric_name

    activity = d.copy()
    activity["match_date"] = pd.to_datetime(activity["match_date"], errors="coerce", utc=True)
    activity = activity.loc[activity["match_date"].notna()].copy()
    if not activity.empty:
        activity["is_recent_365d"] = activity["match_date"].ge(
            reference_date - pd.Timedelta(days=365)
        )
        activity["recent_delivery_units_365d"] = (
            activity["is_recent_365d"].astype(float) * activity["delivery_unit"].astype(float)
        )
        activity_summary = activity.groupby("player_id").agg(
            last_pressure_match_date=("match_date", "max"),
            recent_pressure_deliveries_365d=("recent_delivery_units_365d", "sum"),
            recent_pressure_matches_365d=(
                "match_id",
                lambda s: s.loc[activity.loc[s.index, "is_recent_365d"]].nunique(),
            ),
        ).reset_index()
        activity_summary["days_since_last_pressure_match"] = (
            reference_date - activity_summary["last_pressure_match_date"]
        ).dt.total_seconds() / 86400.0
        grouped = grouped.merge(
            activity_summary, on="player_id", how="left", validate="one_to_one"
        )
    grouped["role"] = role
    return grouped

def build_player_rankings(
    real: pd.DataFrame,
    min_deliveries: int = DEFAULT_MIN_DELIVERIES,
    min_matches: int = DEFAULT_MIN_MATCHES,
    prior_deliveries: int = DEFAULT_PRIOR_DELIVERIES,
    high_pressure_threshold: float = 60.0,
    half_lives_days: tuple[int, ...] = RECENCY_HALF_LIVES_DAYS,
    reference_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Build role-correct pressure rankings with personal-baseline uplift.

    Batters are evaluated when the batting side is under pressure. Bowlers are
    evaluated when the defending side is under pressure (the inverse chase
    state). A low-pressure chase baseline is computed for each player using
    the same PI/XG method and recency horizon.
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
    if reference_date is None:
        reference = pd.Timestamp.now(tz="UTC").normalize()
    else:
        reference = pd.to_datetime(reference_date, errors="raise", utc=True)
        reference = pd.Timestamp(reference).normalize()
    date_reference = reference.isoformat()

    recency_specs: list[tuple[str, int | None]] = [("unweighted", None)]
    recency_specs.extend((f"{h}d", h) for h in half_lives_days)
    frames = []

    for method, score_col in SCORE_METHODS.items():
        if score_col not in real.columns:
            continue
        scores = pd.to_numeric(real[score_col], errors="coerce")
        for role in ("Batter", "Bowler"):
            if role == "Batter":
                pressure_rows = real.loc[scores.ge(high_pressure_threshold)].copy()
                baseline_rows = real.loc[scores.lt(100.0 - high_pressure_threshold)].copy()
            else:
                # Invert batter-side pressure: a bowler is under pressure when the
                # chasing batting side is relatively comfortable / in control.
                pressure_rows = real.loc[scores.le(100.0 - high_pressure_threshold)].copy()
                baseline_rows = real.loc[scores.gt(high_pressure_threshold)].copy()
            if pressure_rows.empty:
                continue
            pressure_rows["match_date"] = pd.to_datetime(
                pressure_rows["match_date"], errors="coerce", utc=True
            )
            baseline_rows["match_date"] = pd.to_datetime(
                baseline_rows["match_date"], errors="coerce", utc=True
            )

            for half_life_label, half_life_days in recency_specs:
                def apply_recency(part: pd.DataFrame) -> pd.DataFrame:
                    result = part.copy()
                    if half_life_days is None:
                        result["recency_weight"] = 1.0
                    else:
                        age_days = (
                            reference - result["match_date"]
                        ).dt.total_seconds() / 86400.0
                        result["recency_weight"] = np.power(
                            2.0, -age_days.clip(lower=0) / float(half_life_days)
                        )
                        result.loc[result["match_date"].isna(), "recency_weight"] = np.nan
                        result = result.loc[result["recency_weight"].notna()].copy()
                    return result

                high_part = apply_recency(pressure_rows)
                base_part = apply_recency(baseline_rows)
                summary = _summarise_role(high_part, role, prior_deliveries, reference)
                if summary.empty:
                    continue
                baseline = _summarise_role(base_part, role, prior_deliveries, reference)
                if not baseline.empty:
                    rename = {
                        "pressure_deliveries": "baseline_deliveries",
                        "pressure_matches": "baseline_matches",
                        "shrunk_per_100": "baseline_shrunk_per_100",
                        "weighted_per_100": "baseline_weighted_per_100",
                        "observed_per_100": "baseline_observed_per_100",
                        "economy_per_over": "baseline_economy_per_over",
                        "dot_rate": "baseline_dot_rate",
                    }
                    keep = ["player_id"] + [c for c in rename if c in baseline.columns]
                    baseline = baseline[keep].rename(columns=rename)
                    summary = summary.merge(baseline, on="player_id", how="left", validate="one_to_one")
                else:
                    for col in ["baseline_deliveries", "baseline_matches", "baseline_shrunk_per_100",
                                "baseline_weighted_per_100", "baseline_observed_per_100",
                                "baseline_economy_per_over", "baseline_dot_rate"]:
                        summary[col] = np.nan

                summary["pressure_uplift_per_100"] = (
                    summary["shrunk_per_100"] - summary["baseline_shrunk_per_100"]
                )
                if role == "Batter":
                    rate_pct = summary["shrunk_per_100"].rank(pct=True)
                    uplift_pct = summary["pressure_uplift_per_100"].rank(pct=True)
                    uplift_pct = uplift_pct.where(summary["pressure_uplift_per_100"].notna(), rate_pct)
                    summary["selection_score"] = 100.0 * (.75 * rate_pct + .25 * uplift_pct)
                else:
                    wicket_pct = summary["shrunk_per_100"].rank(pct=True)
                    economy_pct = summary["economy_per_over"].rank(pct=True, ascending=False)
                    dot_pct = summary["dot_rate"].rank(pct=True)
                    uplift_pct = summary["pressure_uplift_per_100"].rank(pct=True)
                    uplift_pct = uplift_pct.where(summary["pressure_uplift_per_100"].notna(), wicket_pct)
                    summary["selection_score"] = 100.0 * (
                        .45 * wicket_pct + .20 * economy_pct + .15 * dot_pct + .20 * uplift_pct
                    )

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
                    summary.loc[eligible_idx, "selection_score"]
                    .rank(method="min", ascending=False)
                )
                frames.append(summary)

    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values(
            ["role", "scoring_method", "half_life_days", "eligible", "rank", "selection_score"],
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


# Weighted Reciprocal Rank Fusion settings. These are intentionally not equal:
# the transparent PI baseline gets the largest vote; its sensitivity variants
# get progressively smaller votes; synthetic-trained XGBoost gets only 5%
# because its labels come from a constructed teacher rule, not empirical
# pressure ground truth.
DEFAULT_RRF_WEIGHTS = {
    "PI baseline": 0.35,
    "PI RRR-heavy": 0.25,
    "PI momentum-heavy": 0.20,
    "PI stakes-heavy": 0.15,
    "XGBoost": 0.05,
}
DEFAULT_RRF_K = 60
DEFAULT_RRF_MIN_MODEL_SUPPORT = 3


def build_weighted_rrf_ensemble(
    rankings: pd.DataFrame,
    min_deliveries: int = DEFAULT_MIN_DELIVERIES,
    min_matches: int = DEFAULT_MIN_MATCHES,
    half_life: str | None = "180d",
    max_days_since_last_match: int | None = 365,
    min_model_support: int = DEFAULT_RRF_MIN_MODEL_SUPPORT,
    rrf_k: int = DEFAULT_RRF_K,
    method_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build an eligibility-aware, weighted Reciprocal Rank Fusion shortlist.

    Each scoring method is ranked independently after applying the current
    sample-size and activity gates. A player's RRF score is the weighted sum
    of 1 / (rrf_k + model_rank). Missing model ranks contribute zero, and a
    minimum number of model appearances is required before a player can enter
    the ensemble. No final Top-10 decision is made here.
    """
    if rankings.empty:
        return pd.DataFrame()
    if min_deliveries < 0 or min_matches < 0:
        raise ValueError("Minimum deliveries and matches cannot be negative")
    if min_model_support < 1:
        raise ValueError("min_model_support must be at least 1")
    if rrf_k < 0:
        raise ValueError("rrf_k cannot be negative")

    weights = dict(method_weights or DEFAULT_RRF_WEIGHTS)
    if not weights:
        raise ValueError("At least one scoring-method weight is required")
    if any((not np.isfinite(v)) or v <= 0 for v in weights.values()):
        raise ValueError("RRF method weights must be finite and positive")
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-8):
        raise ValueError("RRF method weights must sum to 1.0")
    missing_columns = {"role", "scoring_method", "half_life", "player_id",
                       "player_name", "pressure_deliveries", "pressure_matches",
                       "shrunk_per_100"} - set(rankings.columns)
    if missing_columns:
        raise ValueError(f"Rankings are missing RRF columns: {sorted(missing_columns)}")

    data = rankings.copy()
    if half_life is not None:
        data = data.loc[data["half_life"].eq(half_life)].copy()
    if data.empty:
        return pd.DataFrame()

    output_frames = []
    half_lives = sorted(data["half_life"].dropna().astype(str).unique())
    for half in half_lives:
        half_data = data.loc[data["half_life"].eq(half)]
        for role in ("Batter", "Bowler"):
            role_data = half_data.loc[half_data["role"].eq(role)].copy()
            if role_data.empty:
                continue

            method_frames = []
            for method, weight in weights.items():
                candidates = role_data.loc[role_data["scoring_method"].eq(method)].copy()
                if candidates.empty:
                    continue
                eligible = (
                    pd.to_numeric(candidates["pressure_deliveries"], errors="coerce").ge(min_deliveries)
                    & pd.to_numeric(candidates["pressure_matches"], errors="coerce").ge(min_matches)
                    & pd.to_numeric(candidates["player_id"], errors="coerce").notna()
                    & pd.to_numeric(candidates["shrunk_per_100"], errors="coerce").notna()
                )
                if max_days_since_last_match is not None and "days_since_last_pressure_match" in candidates:
                    eligible &= pd.to_numeric(
                        candidates["days_since_last_pressure_match"], errors="coerce"
                    ).le(max_days_since_last_match)
                candidates = candidates.loc[eligible].copy()
                if candidates.empty:
                    continue

                # One row per player ID per model; names are descriptive, IDs
                # are the stable key used to aggregate rankings across models.
                candidates["player_id"] = pd.to_numeric(candidates["player_id"], errors="coerce")
                rank_col = "selection_score" if "selection_score" in candidates and candidates["selection_score"].notna().any() else "shrunk_per_100"
                candidates = candidates.sort_values(
                    [rank_col, "player_id"],
                    ascending=[False, True],
                    kind="mergesort",
                ).drop_duplicates("player_id", keep="first")
                candidates["model_rank"] = candidates[rank_col].rank(
                    method="min", ascending=False
                ).astype(int)
                candidates["model_weight"] = float(weight)
                candidates["rrf_contribution"] = (
                    float(weight) / (float(rrf_k) + candidates["model_rank"])
                )
                method_frames.append(candidates[[
                    "player_id", "player_name", "scoring_method", "model_rank",
                    "model_weight", "rrf_contribution", "pressure_deliveries",
                    "pressure_matches", "last_pressure_match_date",
                    "days_since_last_pressure_match",
                    *[c for c in ["selection_score", "pressure_uplift_per_100", "economy_per_over", "dot_rate"] if c in candidates.columns],
                ]].copy())

            if not method_frames:
                continue
            long = pd.concat(method_frames, ignore_index=True)
            summary = long.groupby("player_id", as_index=False).agg(
                player_name=("player_name", "first"),
                rrf_score=("rrf_contribution", "sum"),
                model_support=("scoring_method", "nunique"),
                model_weight_coverage=("model_weight", "sum"),
                min_pressure_deliveries_supported=("pressure_deliveries", "min"),
                max_pressure_deliveries_supported=("pressure_deliveries", "max"),
                min_pressure_matches_supported=("pressure_matches", "min"),
                max_pressure_matches_supported=("pressure_matches", "max"),
                last_pressure_match_date=("last_pressure_match_date", "max"),
                days_since_last_pressure_match=("days_since_last_pressure_match", "min"),
                mean_selection_score=("selection_score", "mean") if "selection_score" in long else ("model_rank", "mean"),
                mean_pressure_uplift_per_100=("pressure_uplift_per_100", "mean") if "pressure_uplift_per_100" in long else ("model_rank", "mean"),
            )

            for method, _weight in weights.items():
                slug = method.lower().replace(" ", "_").replace("-", "_")
                method_rows = long.loc[long["scoring_method"].eq(method)].drop_duplicates("player_id").set_index("player_id")
                if not method_rows.empty:
                    summary = summary.join(
                        method_rows[[c for c in ["model_rank", "rrf_contribution", "selection_score", "pressure_uplift_per_100", "economy_per_over", "dot_rate"] if c in method_rows]].rename(columns={
                            "model_rank": f"rank_{slug}",
                            "rrf_contribution": f"contribution_{slug}",
                            "selection_score": f"selection_score_{slug}",
                            "pressure_uplift_per_100": f"pressure_uplift_per_100_{slug}",
                            "economy_per_over": f"economy_per_over_{slug}",
                            "dot_rate": f"dot_rate_{slug}",
                        }),
                        on="player_id",
                    )
                else:
                    summary[f"rank_{slug}"] = np.nan
                    summary[f"contribution_{slug}"] = 0.0

            summary = summary.loc[summary["model_support"].ge(min_model_support)].copy()
            if summary.empty:
                continue
            summary["role"] = role
            summary["half_life"] = half
            summary["min_deliveries"] = int(min_deliveries)
            summary["min_matches"] = int(min_matches)
            summary["max_days_since_last_match"] = (
                int(max_days_since_last_match)
                if max_days_since_last_match is not None else np.nan
            )
            summary["min_model_support"] = int(min_model_support)
            summary["rrf_k"] = int(rrf_k)
            summary["model_weights"] = "; ".join(
                f"{name}={weight:.0%}" for name, weight in weights.items()
            )
            ref_dates = (
                role_data["recency_reference_date"].dropna().astype(str).unique()
                if "recency_reference_date" in role_data else []
            )
            summary["recency_reference_date"] = ref_dates[0] if len(ref_dates) else None
            summary = summary.sort_values(
                ["rrf_score", "model_support", "model_weight_coverage", "player_id"],
                ascending=[False, False, False, True],
                kind="mergesort",
            ).reset_index(drop=True)
            summary["ensemble_rank"] = np.arange(1, len(summary) + 1)
            output_frames.append(summary)

    if not output_frames:
        return pd.DataFrame()
    return pd.concat(output_frames, ignore_index=True, sort=False).sort_values(
        ["role", "half_life", "ensemble_rank"]
    ).reset_index(drop=True)


def compare_top_rrf_ensemble(ensemble: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return the first N eligible players from each role/half-life ensemble."""
    if ensemble.empty:
        return pd.DataFrame()
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    return ensemble.loc[ensemble["ensemble_rank"].le(top_n)].copy()
