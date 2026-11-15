from __future__ import annotations

import numpy as np
import pandas as pd

# A phase-aware PI is more useful for role matching than one fixed pressure mix.
# The chosen phase PI is used for both RR diagnostics and all-IPL candidate ranks.
PI_WEIGHTS = {
    "baseline": {
        "rrr_pressure": .25, "wicket_pressure": .20, "time_urgency": .15,
        "momentum_pressure": .15, "death_over_pressure": .10,
        "target_difficulty": .075, "match_stakes": .075,
    },
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
PHASE_PI = {
    "Powerplay": "baseline",
    "Middle": "momentum_heavy",
    "Death": "rrr_situation_heavy",
}
PHASE_ROLE_NEED = {
    ("Batter", "Powerplay"): "Top-order aggressor: score quickly without losing early wickets",
    ("Batter", "Middle"): "Middle-overs controller: rotate strike and convert scoring balls into boundaries",
    ("Batter", "Death"): "Death-overs finisher: high strike rate, boundary conversion and dismissal control in chases",
    ("Bowler", "Powerplay"): "New-ball wicket-taker: control economy while creating early breakthroughs",
    ("Bowler", "Middle"): "Middle-overs controller: suppress scoring through dots and take wickets",
    ("Bowler", "Death"): "Death specialist: defend the final overs with low economy, dots and wicket threat",
}


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def _key(frame: pd.DataFrame, id_col: str, name_col: str) -> pd.Series:
    ids = frame[id_col].astype("string") if id_col in frame else pd.Series(pd.NA, index=frame.index, dtype="string")
    names = frame[name_col].fillna("Unknown").astype(str) if name_col in frame else pd.Series("Unknown", index=frame.index)
    return ids.where(ids.notna() & ids.ne(""), names).astype(str)


def _add_phase_pi(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    missing = set(next(iter(PI_WEIGHTS.values())).keys()) - set(d.columns)
    if missing:
        raise ValueError(f"RR delivery output is missing PI components: {sorted(missing)}")
    eligible = _as_bool(d.get("chase_active", pd.Series(False, index=d.index)))
    for name, weights in PI_WEIGHTS.items():
        col = f"pi_{name}_score"
        d[col] = sum(pd.to_numeric(d[k], errors="coerce").fillna(0) * weight for k, weight in weights.items()) * 100
        d.loc[~eligible, col] = np.nan
    phase_score = pd.Series(np.nan, index=d.index, dtype=float)
    for phase, pi_name in PHASE_PI.items():
        phase_score.loc[d["phase"].eq(phase)] = d.loc[d["phase"].eq(phase), f"pi_{pi_name}_score"]
    d["phase_pi_type"] = d["phase"].map(PHASE_PI)
    d["phase_pi_score"] = phase_score
    d["phase_bowling_pi_score"] = 100 - phase_score
    d["phase_pi_high"] = d["phase_pi_score"].ge(60)
    d["phase_bowling_pi_high"] = d["phase_bowling_pi_score"].ge(60)
    d["phase_pi_baseline"] = d["phase_pi_score"].lt(40)
    d["phase_bowling_pi_baseline"] = d["phase_bowling_pi_score"].lt(40)
    xgb = pd.to_numeric(d.get("xgb_pressure_score"), errors="coerce")
    d["xgb_bowling_pressure_score"] = 100 - xgb
    d["xgb_batting_high"] = xgb.ge(60)
    d["xgb_bowling_high"] = d["xgb_bowling_pressure_score"].ge(60)
    return d


def _player_metrics(frame: pd.DataFrame, role: str, pressure_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    d = frame.copy()
    if role == "Batter":
        if "batter_ball_faced" in d:
            d = d.loc[_as_bool(d["batter_ball_faced"])].copy()
        else:
            d = d.loc[~_as_bool(d.get("is_wide", pd.Series(False, index=d.index)))].copy()
        if d.empty:
            return pd.DataFrame()
        d["player_key"] = _key(d, "batter_id", "batter")
        d["player_name"] = d["batter"].fillna("Unknown")
        d["runs_scored"] = pd.to_numeric(d["runs_batter"], errors="coerce").fillna(0)
        d["dismissal"] = _as_bool(d["dismissed_striker"]) if "dismissed_striker" in d else False
        d["boundary"] = _as_bool(d["is_boundary_four"]) | _as_bool(d["is_boundary_six"])
        d["dot"] = d["runs_scored"].eq(0)
        result = d.groupby(["player_key", "player_name"], as_index=False).agg(
            deliveries=("match_id", "size"), matches=("match_id", "nunique"),
            runs=("runs_scored", "sum"), dismissals=("dismissal", "sum"),
            boundaries=("boundary", "sum"), dots=("dot", "sum"),
            mean_pressure=(pressure_col, "mean"),
        )
        result["strike_rate"] = 100 * result["runs"] / result["deliveries"].replace(0, np.nan)
        result["boundary_rate"] = 100 * result["boundaries"] / result["deliveries"].replace(0, np.nan)
        result["dismissals_per_100"] = 100 * result["dismissals"] / result["deliveries"].replace(0, np.nan)
        result["dot_rate"] = 100 * result["dots"] / result["deliveries"].replace(0, np.nan)
        return result

    if "is_legal_ball" in d:
        d = d.loc[_as_bool(d["is_legal_ball"])].copy()
    if d.empty:
        return pd.DataFrame()
    d["player_key"] = _key(d, "bowler_id", "bowler")
    d["player_name"] = d["bowler"].fillna("Unknown")
    d["runs_conceded"] = (
        pd.to_numeric(d["runs_batter"], errors="coerce").fillna(0)
        + pd.to_numeric(d["wides"], errors="coerce").fillna(0)
        + pd.to_numeric(d["no_balls"], errors="coerce").fillna(0)
    )
    d["credited_wicket"] = _as_bool(d["bowler_credited_wicket"]) if "bowler_credited_wicket" in d else False
    d["dot"] = d["runs_conceded"].eq(0)
    result = d.groupby(["player_key", "player_name"], as_index=False).agg(
        deliveries=("match_id", "size"), matches=("match_id", "nunique"),
        runs_conceded=("runs_conceded", "sum"), wickets=("credited_wicket", "sum"),
        dots=("dot", "sum"), mean_pressure=(pressure_col, "mean"),
    )
    result["economy"] = 6 * result["runs_conceded"] / result["deliveries"].replace(0, np.nan)
    result["wickets_per_100"] = 100 * result["wickets"] / result["deliveries"].replace(0, np.nan)
    result["dot_rate"] = 100 * result["dots"] / result["deliveries"].replace(0, np.nan)
    return result


def _rr_player_phase_record(rr_rows: pd.DataFrame, role: str, phase: str, rr_team: str) -> pd.DataFrame:
    d = rr_rows.loc[rr_rows["phase"].eq(phase) & _as_bool(rr_rows["chase_active"])].copy()
    if role == "Batter":
        d = d.loc[d["batting_team"].eq(rr_team) & _as_bool(d["batter_ball_faced"])].copy()
        id_col, name_col = "batter_id", "batter"
        pi_high, pi_base, xgb_high = "phase_pi_high", "phase_pi_baseline", "xgb_batting_high"
    else:
        d = d.loc[d["bowling_team"].eq(rr_team) & _as_bool(d["is_legal_ball"])].copy()
        id_col, name_col = "bowler_id", "bowler"
        pi_high, pi_base, xgb_high = "phase_bowling_pi_high", "phase_bowling_pi_baseline", "xgb_bowling_high"
    if d.empty:
        return pd.DataFrame()
    d["player_key"] = _key(d, id_col, name_col)
    d["player_name"] = d[name_col].fillna("Unknown")
    pi_rows = d.loc[_as_bool(d[pi_high])].copy()
    base_rows = d.loc[_as_bool(d[pi_base])].copy()
    xgb_rows = d.loc[_as_bool(d[xgb_high])].copy()
    pi_pressure_col = "phase_pi_score" if role == "Batter" else "phase_bowling_pi_score"
    xgb_pressure_col = "xgb_pressure_score" if role == "Batter" else "xgb_bowling_pressure_score"
    pi = _player_metrics(pi_rows, role, pi_pressure_col)
    base = _player_metrics(base_rows, role, pi_pressure_col)
    xgb = _player_metrics(xgb_rows, role, xgb_pressure_col)

    if not pi.empty:
        pi = pi.rename(columns={
            "deliveries": "rr_pi_pressure_deliveries", "matches": "rr_pi_pressure_matches",
            "strike_rate": "rr_pi_pressure_strike_rate", "boundary_rate": "rr_pi_pressure_boundary_rate",
            "dismissals_per_100": "rr_pi_pressure_dismissals_per_100", "economy": "rr_pi_pressure_economy",
            "wickets_per_100": "rr_pi_pressure_wickets_per_100", "dot_rate": "rr_pi_pressure_dot_rate",
        }).drop(columns=["runs", "dismissals", "boundaries", "dots", "runs_conceded", "wickets", "mean_pressure"], errors="ignore")
    if not base.empty:
        base = base.rename(columns={
            "deliveries": "rr_baseline_deliveries", "matches": "rr_baseline_matches",
            "strike_rate": "rr_baseline_strike_rate", "boundary_rate": "rr_baseline_boundary_rate",
            "dismissals_per_100": "rr_baseline_dismissals_per_100", "economy": "rr_baseline_economy",
            "wickets_per_100": "rr_baseline_wickets_per_100", "dot_rate": "rr_baseline_dot_rate",
        }).drop(columns=["runs", "dismissals", "boundaries", "dots", "runs_conceded", "wickets", "mean_pressure"], errors="ignore")
    if not xgb.empty:
        xgb = xgb.rename(columns={
            "deliveries": "rr_xgb_pressure_deliveries", "matches": "rr_xgb_pressure_matches",
            "strike_rate": "rr_xgb_pressure_strike_rate", "economy": "rr_xgb_pressure_economy",
            "wickets_per_100": "rr_xgb_pressure_wickets_per_100", "dot_rate": "rr_xgb_pressure_dot_rate",
        }).drop(columns=["runs", "dismissals", "boundaries", "dots", "runs_conceded", "wickets", "mean_pressure", "boundary_rate", "dismissals_per_100"], errors="ignore")

    keys = d[["player_key", "player_name"]].drop_duplicates()
    result = keys.merge(pi, on=["player_key", "player_name"], how="left")
    if not base.empty:
        result = result.merge(base, on=["player_key", "player_name"], how="left")
    if not xgb.empty:
        result = result.merge(xgb, on=["player_key", "player_name"], how="left")
    expected = (
        ["rr_pi_pressure_deliveries", "rr_pi_pressure_matches", "rr_pi_pressure_strike_rate",
         "rr_pi_pressure_boundary_rate", "rr_pi_pressure_dismissals_per_100",
         "rr_baseline_deliveries", "rr_baseline_matches", "rr_baseline_strike_rate",
         "rr_baseline_boundary_rate", "rr_baseline_dismissals_per_100",
         "rr_xgb_pressure_deliveries", "rr_xgb_pressure_matches", "rr_xgb_pressure_strike_rate"]
        if role == "Batter" else
        ["rr_pi_pressure_deliveries", "rr_pi_pressure_matches", "rr_pi_pressure_economy",
         "rr_pi_pressure_wickets_per_100", "rr_pi_pressure_dot_rate",
         "rr_baseline_deliveries", "rr_baseline_matches", "rr_baseline_economy",
         "rr_baseline_wickets_per_100", "rr_baseline_dot_rate",
         "rr_xgb_pressure_deliveries", "rr_xgb_pressure_matches", "rr_xgb_pressure_economy"]
    )
    for col in expected:
        if col not in result:
            result[col] = np.nan

    if role == "Batter":
        result["pressure_sr_uplift"] = result["rr_pi_pressure_strike_rate"] - result["rr_baseline_strike_rate"]
        result["pressure_boundary_uplift"] = result["rr_pi_pressure_boundary_rate"] - result["rr_baseline_boundary_rate"]
        result["pressure_dismissal_delta"] = result["rr_pi_pressure_dismissals_per_100"] - result["rr_baseline_dismissals_per_100"]
        underperformance = (
            .50 * ((-result["pressure_sr_uplift"]).clip(lower=0) / 15).clip(0, 1)
            + .25 * ((-result["pressure_boundary_uplift"]).clip(lower=0) / 5).clip(0, 1)
            + .25 * (result["pressure_dismissal_delta"].clip(lower=0) / 1.5).clip(0, 1)
        )
    else:
        result["pressure_economy_delta"] = result["rr_pi_pressure_economy"] - result["rr_baseline_economy"]
        result["pressure_wicket_delta"] = result["rr_pi_pressure_wickets_per_100"] - result["rr_baseline_wickets_per_100"]
        result["pressure_dot_delta"] = result["rr_pi_pressure_dot_rate"] - result["rr_baseline_dot_rate"]
        underperformance = (
            .45 * (result["pressure_economy_delta"].clip(lower=0) / 2).clip(0, 1)
            + .35 * ((-result["pressure_wicket_delta"]).clip(lower=0) / 2).clip(0, 1)
            + .20 * ((-result["pressure_dot_delta"]).clip(lower=0) / 5).clip(0, 1)
        )
    result["rr_pressure_underperformance_score"] = underperformance * 100
    result["rr_pressure_sample_adequate"] = (
        pd.to_numeric(result["rr_pi_pressure_deliveries"], errors="coerce").fillna(0).ge(12)
        & pd.to_numeric(result["rr_pi_pressure_matches"], errors="coerce").fillna(0).ge(2)
    )
    result["role"] = role
    result["phase"] = phase
    return result


def build_rr_weakness_replacements(
    rr_deliveries: pd.DataFrame,
    frequent_xi: pd.DataFrame,
    phase_gaps: pd.DataFrame,
    phase_rankings: pd.DataFrame,
    rr_fit: pd.DataFrame,
    rr_team: str = "Rajasthan Royals",
    min_candidate_pressure_balls: int = 20,
    min_candidate_pressure_matches: int = 3,
    top_candidates: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify RR XI role/phase weaknesses and rank like-for-like replacements."""
    if rr_deliveries.empty or frequent_xi.empty:
        return pd.DataFrame(), pd.DataFrame()
    rr = _add_phase_pi(rr_deliveries)
    xi = frequent_xi.copy()
    xi["player_key"] = xi["player_id"].astype(str)

    role_rows = []
    for _, player in xi.iterrows():
        roles = []
        if pd.to_numeric(player.get("rr_batting_balls", 0), errors="coerce") > 0:
            roles.append("Batter")
        if pd.to_numeric(player.get("rr_bowling_legal_balls", 0), errors="coerce") > 0:
            roles.append("Bowler")
        for role in roles:
            for phase in ("Powerplay", "Middle", "Death"):
                role_rows.append({
                    "player_key": str(player["player_id"]), "player_name": player["player_name"],
                    "historical_xi_rank": player.get("historical_xi_rank"),
                    "appearances": player.get("appearances"), "role": role, "phase": phase,
                    "usage_profile": player.get("usage_profile"),
                })
    if not role_rows:
        return pd.DataFrame(), pd.DataFrame()
    xi_roles = pd.DataFrame(role_rows)

    rr_metrics = []
    for role in ("Batter", "Bowler"):
        for phase in ("Powerplay", "Middle", "Death"):
            m = _rr_player_phase_record(rr, role, phase, rr_team)
            if not m.empty:
                rr_metrics.append(m)
    rr_metrics = pd.concat(rr_metrics, ignore_index=True) if rr_metrics else pd.DataFrame()
    review = xi_roles.merge(rr_metrics, on=["player_key", "player_name", "role", "phase"], how="left")

    if not phase_gaps.empty:
        gaps = phase_gaps.rename(columns={"role": "gap_role"}).copy()
        gaps["role"] = gaps["gap_role"].map({"Batting": "Batter", "Bowling": "Bowler"})
        gap_cols = [c for c in [
            "role", "phase", "team_need_score", "rr_balls", "rr_strike_rate", "league_strike_rate",
            "rr_economy", "league_economy", "rr_wickets_per_100_balls",
            "league_wickets_per_100_balls", "rr_dot_rate", "league_dot_rate",
        ] if c in gaps]
        review = review.merge(
            gaps[gap_cols].drop_duplicates(["role", "phase"]),
            on=["role", "phase"], how="left",
        )

    if not phase_rankings.empty:
        model_cols = [c for c in [
            "player_key", "role", "phase", "decision_score", "decision_rank",
            "pi_performance_score", "xgb_performance_score", "pi_type",
            "model_weight_coverage", "pi_pressure_deliveries", "pi_pressure_matches",
            "xgb_pressure_deliveries", "xgb_pressure_matches", "phase_deliveries", "phase_matches",
        ] if c in phase_rankings]
        model_frame = phase_rankings[model_cols].copy()
        model_frame["player_key"] = model_frame["player_key"].astype(str)
        review = review.merge(
            model_frame.drop_duplicates(["player_key", "role", "phase"]),
            on=["player_key", "role", "phase"], how="left",
        )

    if not rr_fit.empty:
        fit = rr_fit.copy()
        fit["player_key"] = fit["player_id"].astype(str)
        fit_cols = [c for c in [
            "player_key", "role", "phase", "player_fit_score", "auction_fit_score",
            "pressure_rate", "pressure_deliveries", "pressure_matches",
            "strike_rate_or_economy", "boundary_rate", "dot_rate", "wicket_rate", "franchises",
        ] if c in fit]
        fit = fit[fit_cols].drop_duplicates(["player_key", "role", "phase"])
        review = review.merge(fit, on=["player_key", "role", "phase"], how="left")
    else:
        review["player_fit_score"] = np.nan
        review["auction_fit_score"] = np.nan

    if "team_need_score" not in review:
        review["team_need_score"] = np.nan
    if "decision_score" not in review:
        review["decision_score"] = np.nan
    review["decision_score"] = pd.to_numeric(review["decision_score"], errors="coerce")
    review["team_need_score"] = pd.to_numeric(review["team_need_score"], errors="coerce").fillna(0)
    review["rr_pressure_underperformance_score"] = pd.to_numeric(
        review.get("rr_pressure_underperformance_score"), errors="coerce"
    )
    review["model_weakness_component"] = (100 - review["decision_score"]).clip(0, 100).fillna(50)
    review["underperformance_component"] = review["rr_pressure_underperformance_score"].fillna(50)
    review["weakness_score"] = (
        .35 * review["team_need_score"]
        + .40 * review["model_weakness_component"]
        + .25 * review["underperformance_component"]
    ).clip(0, 100)

    adequate = _as_bool(review.get("rr_pressure_sample_adequate", pd.Series(False, index=review.index)))
    pressure_drop = review["rr_pressure_underperformance_score"].ge(50)
    model_low = review["decision_score"].lt(45)
    team_need = review["team_need_score"].ge(35)
    review["weakness_status"] = np.select(
        [adequate & (pressure_drop | (model_low & team_need)), ~adequate & team_need],
        ["Pressure weakness supported by RR sample", "Squad need — individual RR pressure sample is limited"],
        default="No strong weakness signal",
    )
    review["role_requirement"] = [
        PHASE_ROLE_NEED.get((r, p), "Role/phase fit")
        for r, p in zip(review["role"], review["phase"])
    ]

    reasons = []
    for _, row in review.iterrows():
        parts = [f"{row['phase']} {row['role'].lower()} need score {row['team_need_score']:.1f}/100"]
        if row["role"] == "Batter" and pd.notna(row.get("pressure_sr_uplift")):
            direction = "falls" if row["pressure_sr_uplift"] < 0 else "rises"
            parts.append(
                f"RR chase pressure strike rate {direction} {abs(row['pressure_sr_uplift']):.1f} points "
                "vs the player's lower-pressure chase baseline"
            )
            if pd.notna(row.get("pressure_dismissal_delta")):
                parts.append(f"dismissal-rate change {row['pressure_dismissal_delta']:+.2f} per 100 balls")
        elif row["role"] == "Bowler" and pd.notna(row.get("pressure_economy_delta")):
            direction = "worsens" if row["pressure_economy_delta"] > 0 else "improves"
            parts.append(
                f"RR chase pressure economy {direction} by {abs(row['pressure_economy_delta']):.2f} runs/over "
                "vs baseline"
            )
            if pd.notna(row.get("pressure_wicket_delta")):
                parts.append(f"wicket-rate change {row['pressure_wicket_delta']:+.2f} per 100 balls")
        else:
            parts.append("insufficient RR high-pressure deliveries to claim a measured pressure failure")
        reasons.append("; ".join(parts))
    review["weakness_reason"] = reasons
    review = review.sort_values(["weakness_score", "team_need_score"], ascending=False).reset_index(drop=True)

    # Replacement pool is the wider IPL player pool, not the RR-only XI. Exclude every
    # current XI member and retain only same-role, same-phase candidates.
    replacements = []
    current_ids = set(xi["player_key"].astype(str))
    fit_candidates = rr_fit.copy() if not rr_fit.empty else pd.DataFrame()
    if not fit_candidates.empty:
        fit_candidates["player_key"] = fit_candidates["player_id"].astype(str)
        fit_candidates["player_fit_score"] = pd.to_numeric(fit_candidates.get("player_fit_score"), errors="coerce")
    weak_rows = review.loc[review["weakness_status"].ne("No strong weakness signal")].copy()
    if fit_candidates.empty or phase_rankings.empty:
        return review, pd.DataFrame()

    for _, weak in weak_rows.iterrows():
        pool = phase_rankings.loc[
            phase_rankings["role"].eq(weak["role"]) & phase_rankings["phase"].eq(weak["phase"])
        ].copy()
        pool["player_key"] = pool["player_key"].astype(str)
        pool = pool.loc[~pool["player_key"].isin(current_ids)]
        pool = pool.merge(
            fit_candidates.drop(columns=["player_id"], errors="ignore"),
            on=["player_key", "role", "phase"], how="left", suffixes=("", "_fit"),
        )
        if pool.empty:
            continue

        pressure_balls = pd.to_numeric(pool.get("pi_pressure_deliveries"), errors="coerce").fillna(0)
        pressure_matches = pd.to_numeric(pool.get("pi_pressure_matches"), errors="coerce").fillna(0)
        gated = pool.loc[
            pressure_balls.ge(min_candidate_pressure_balls)
            & pressure_matches.ge(min_candidate_pressure_matches)
            & pd.to_numeric(pool["decision_score"], errors="coerce").notna()
        ].copy()
        if gated.empty:
            gated = pool.loc[
                pd.to_numeric(pool["phase_deliveries"], errors="coerce").fillna(0).ge(30)
                & pd.to_numeric(pool["decision_score"], errors="coerce").notna()
            ].copy()
            gated["candidate_sample_status"] = "Relaxed pressure-sample gate — validate before selection"
        else:
            gated["candidate_sample_status"] = "Meets PI pressure sample gate"
        if gated.empty:
            continue

        gated["model_score"] = pd.to_numeric(gated["decision_score"], errors="coerce")
        fit_score = pd.to_numeric(gated.get("player_fit_score"), errors="coerce")
        gated["candidate_score"] = (
            .85 * gated["model_score"] + .15 * fit_score.fillna(gated["model_score"])
        )
        sort_col = "pi_pressure_deliveries" if "pi_pressure_deliveries" in gated else "phase_deliveries"
        gated = gated.sort_values(["candidate_score", sort_col], ascending=False).head(top_candidates).copy()
        gated.insert(0, "replaces_player", weak["player_name"])
        gated.insert(1, "weak_role", weak["role"])
        gated.insert(2, "weak_phase", weak["phase"])
        gated["role_requirement"] = weak["role_requirement"]
        gated["rr_team_need_score"] = weak["team_need_score"]
        gated["weakness_reason"] = weak["weakness_reason"]
        gated["candidate_fit_reason"] = (
            f"Best {weak['role'].lower()} option for {weak['phase']} using 80/20 PI/XGBoost pressure score, "
            "with phase-fit metrics as a secondary check"
        )
        replacements.append(gated)

    if not replacements:
        return review, pd.DataFrame()
    result = pd.concat(replacements, ignore_index=True)
    result["replacement_rank"] = result.groupby(
        ["replaces_player", "weak_role", "weak_phase"]
    ).cumcount() + 1
    result["availability_status"] = "Not assessed — verify official 2027 retention/auction pool"
    return review, result