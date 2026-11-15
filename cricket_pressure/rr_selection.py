from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re
import zipfile

import numpy as np
import pandas as pd

from .pressure import add_pressure_features, compute_pressure_index

PI_VARIANT_WEIGHTS = {
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

PHASE_PI_TYPE = {
    "Powerplay": "baseline",
    "Middle": "momentum_heavy",
    "Death": "rrr_situation_heavy",
}
from .synthetic import score_with_xgb

PI_WEIGHT = 0.80
XGB_WEIGHT = 0.20
PRESSURE_CUT = 60.0


def add_project_pressure_scores(frame: pd.DataFrame, xgb_model: Any) -> pd.DataFrame:
    """Apply the exact project PI and saved synthetic-pressure XGBoost to IPL deliveries."""
    if xgb_model is None:
        raise ValueError("Load models/synthetic_pressure_xgb.json before building PI/XG phase rankings.")
    data = frame.copy()
    data["scheduled_balls"] = pd.to_numeric(data["innings_scheduled_overs"], errors="coerce") * 6
    first_totals = data.loc[data["innings_number"].eq(1)].groupby(
        ["match_id", "season"], as_index=False
    ).agg(first_innings_total=("runs_total", "sum"))
    first_totals["target_percentile"] = first_totals.groupby("season")[
        "first_innings_total"
    ].rank(pct=True, method="average")
    data = data.drop(columns=["target_percentile"], errors="ignore").merge(
        first_totals[["match_id", "target_percentile"]],
        on="match_id", how="left", validate="many_to_one",
    )
    data["target_percentile"] = data["target_percentile"].fillna(0.5)
    data["target_difficulty"] = data["target_percentile"]
    data["score_momentum_gap"] = (
        pd.to_numeric(data["required_run_rate"], errors="coerce")
        - pd.to_numeric(data["recent_run_rate_12"], errors="coerce")
    ).clip(lower=0)
    data["chase_active"] = (
        data["innings_number"].eq(2)
        & data["runs_required"].gt(0)
        & data["balls_remaining"].gt(0)
        & ~data["target_is_provisional"].fillna(True)
    )
    data["match_stakes"] = pd.to_numeric(
        data.get("match_stakes", pd.Series(0.25, index=data.index)), errors="coerce"
    ).fillna(0.25).clip(0, 1)
    data["is_t20i"] = 0

    data = add_pressure_features(data)
    data = compute_pressure_index(data)
    eligible = data["pressure_eligible"].fillna(False)

    # Preserve the canonical PI and add sensitivity variants. The phase selector
    # chooses one PI type according to the cricket task: stable baseline in the
    # Powerplay, momentum-sensitive in the Middle, and RRR/time-sensitive at Death.
    for pi_name, weights in PI_VARIANT_WEIGHTS.items():
        score_col = f"pi_{pi_name}_score"
        data[score_col] = sum(data[k].fillna(0) * weight for k, weight in weights.items()) * 100
        data.loc[~eligible, score_col] = np.nan

    data["xgb_pressure_score"] = np.nan
    data["xgb_pressure_class"] = None
    if eligible.any():
        scored = score_with_xgb(xgb_model, data.loc[eligible].copy())
        data.loc[eligible, "xgb_pressure_score"] = scored["xgb_pressure_score"].to_numpy()
        data.loc[eligible, "xgb_pressure_class"] = scored["xgb_pressure_class"].to_numpy()

    # The chase PI describes batting-side pressure. Bowling pressure is its inverse.
    data["bowling_pressure_index"] = 100 - data["pressure_index"]
    data["bowling_xgb_pressure_score"] = 100 - data["xgb_pressure_score"]
    phase_score = pd.Series(np.nan, index=data.index, dtype=float)
    for phase, pi_name in PHASE_PI_TYPE.items():
        phase_mask = data["phase"].eq(phase)
        phase_score.loc[phase_mask] = data.loc[phase_mask, f"pi_{pi_name}_score"]
    data["phase_pi_type"] = data["phase"].map(PHASE_PI_TYPE)
    data["phase_pi_score"] = phase_score
    data["phase_bowling_pi_score"] = 100 - phase_score
    data["is_high_pressure"] = data["pressure_index"].ge(PRESSURE_CUT)
    data["is_xgb_high_pressure"] = data["xgb_pressure_score"].ge(PRESSURE_CUT)
    data["is_bowling_high_pressure"] = data["bowling_pressure_index"].ge(PRESSURE_CUT)
    data["is_bowling_xgb_high_pressure"] = data["bowling_xgb_pressure_score"].ge(PRESSURE_CUT)
    data["is_phase_pi_high_pressure"] = data["phase_pi_score"].ge(PRESSURE_CUT)
    data["is_phase_bowling_pi_high_pressure"] = data["phase_bowling_pi_score"].ge(PRESSURE_CUT)
    return data


def _year(info: dict[str, Any], date: pd.Timestamp) -> int:
    match = re.search(r"(?:19|20)\d{2}", str(info.get("season", "")))
    return int(match.group(0)) if match else int(date.year)


def _append_lineup(rows: list[dict[str, Any]], source_name: str, payload: dict[str, Any],
                   start_year: int, end_year: int, rr_team: str) -> None:
    info = payload.get("info") or {}
    if rr_team not in (info.get("teams") or []):
        return
    dates = info.get("dates") or []
    date = pd.to_datetime(dates[0], utc=True, errors="coerce") if dates else pd.NaT
    if pd.isna(date):
        return
    season = _year(info, date)
    if not start_year <= season <= end_year:
        return
    registry = (info.get("registry") or {}).get("people") or {}
    match_id = Path(source_name).stem
    for name in (info.get("players") or {}).get(rr_team, []) or []:
        rows.append({
            "match_id": match_id, "match_date": date, "season": season,
            "player_name": str(name), "player_id": str(registry.get(name) or name),
        })


def _read_lineups(source: str | Path, start_year: int, end_year: int,
                  rr_team: str) -> pd.DataFrame:
    path = Path(source)
    rows: list[dict[str, Any]] = []
    if path.is_dir():
        for file in sorted(path.rglob("*.json")):
            _append_lineup(rows, str(file), json.loads(file.read_text(encoding="utf-8")),
                           start_year, end_year, rr_team)
    elif path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.lower().endswith(".json") and not name.endswith("/"):
                    _append_lineup(rows, name, json.loads(archive.read(name).decode("utf-8")),
                                   start_year, end_year, rr_team)
    else:
        raise FileNotFoundError(f"Expected Cricsheet ZIP or extracted JSON folder: {path}")
    return pd.DataFrame(rows)


def _key(frame: pd.DataFrame, id_col: str, name_col: str) -> pd.Series:
    ids = frame[id_col].astype("string")
    return ids.where(ids.notna() & ids.ne(""), frame[name_col].fillna("Unknown").astype(str)).astype(str)


def build_frequent_xi(source: str | Path, rr_deliveries: pd.DataFrame,
                       start_year: int, end_year: int,
                       rr_team: str = "Rajasthan Royals") -> pd.DataFrame:
    """Return the 11 most frequently listed RR playing-XI members in the requested season."""
    roster = _read_lineups(source, start_year, end_year, rr_team)
    rr_season = rr_deliveries.loc[
        pd.to_numeric(rr_deliveries["season"], errors="coerce").between(start_year, end_year)
    ].copy()
    if roster.empty:
        # Some source files omit info.players; infer participants only from the requested XI season.
        rows = rr_season.loc[
            rr_season["batting_team"].eq(rr_team) | rr_season["bowling_team"].eq(rr_team)
        ]
        batting = rows.loc[rows["batting_team"].eq(rr_team),
                           ["match_id", "match_date", "season", "batter", "batter_id"]].rename(
            columns={"batter": "player_name", "batter_id": "player_id"})
        bowling = rows.loc[rows["bowling_team"].eq(rr_team),
                           ["match_id", "match_date", "season", "bowler", "bowler_id"]].rename(
            columns={"bowler": "player_name", "bowler_id": "player_id"})
        roster = pd.concat([batting, bowling], ignore_index=True).drop_duplicates(["match_id", "player_name"])
        if roster.empty:
            return pd.DataFrame()
        roster["player_id"] = roster["player_id"].fillna(roster["player_name"]).astype(str)

    appearances = roster.groupby(["player_id", "player_name"], as_index=False).agg(
        appearances=("match_id", "nunique"),
        first_season=("season", "min"), last_season=("season", "max"),
        first_appearance=("match_date", "min"), last_appearance=("match_date", "max"),
    )
    # The displayed XI's role statistics must describe the same season as its XI appearances.
    rr = rr_season.loc[
        rr_season["batting_team"].eq(rr_team) | rr_season["bowling_team"].eq(rr_team)
    ].copy()
    batting = rr.loc[rr["batting_team"].eq(rr_team)].copy()
    batting["batter_key"] = _key(batting, "batter_id", "batter")
    bat_stats = batting.groupby("batter_key", as_index=False).agg(
        rr_batting_matches=("match_id", "nunique"),
        rr_batting_balls=("batter_ball_faced", "sum"),
        rr_batting_runs=("runs_batter", "sum"),
    )
    bowling = rr.loc[rr["bowling_team"].eq(rr_team)].copy()
    bowling["bowler_key"] = _key(bowling, "bowler_id", "bowler")
    bowling["bowler_runs_conceded"] = bowling["runs_batter"] + bowling["wides"] + bowling["no_balls"]
    bowl_stats = bowling.groupby("bowler_key", as_index=False).agg(
        rr_bowling_matches=("match_id", "nunique"),
        rr_bowling_legal_balls=("is_legal_ball", "sum"),
        rr_bowling_runs_conceded=("bowler_runs_conceded", "sum"),
        rr_bowling_wickets=("bowler_credited_wicket", "sum"),
    )
    stats = bat_stats.merge(bowl_stats, left_on="batter_key", right_on="bowler_key", how="outer")
    stats["player_id"] = stats["batter_key"].fillna(stats["bowler_key"])
    result = appearances.merge(stats, on="player_id", how="left")
    numeric = [c for c in result.columns if c.startswith("rr_")]
    result[numeric] = result[numeric].fillna(0)
    result["rr_batting_strike_rate"] = (
        100 * result["rr_batting_runs"] / result["rr_batting_balls"].replace(0, np.nan)
    )
    result["rr_bowling_economy"] = (
        6 * result["rr_bowling_runs_conceded"] / result["rr_bowling_legal_balls"].replace(0, np.nan)
    )
    result["usage_profile"] = np.select(
        [
            result["rr_batting_balls"].gt(0) & result["rr_bowling_legal_balls"].gt(0),
            result["rr_batting_balls"].gt(0),
            result["rr_bowling_legal_balls"].gt(0),
        ],
        ["Batting + bowling", "Primarily batting", "Primarily bowling"],
        default="Playing-XI listing only",
    )
    result = result.sort_values(
        ["appearances", "rr_batting_balls", "rr_bowling_legal_balls", "last_appearance"],
        ascending=[False, False, False, False],
    ).head(11).reset_index(drop=True)
    result.insert(0, "historical_xi_rank", np.arange(1, len(result) + 1))
    result["xi_season"] = int(start_year) if start_year == end_year else f"{start_year}–{end_year}"
    result["xi_scope"] = f"Most frequent RR playing-XI listings, IPL {start_year}" if start_year == end_year else f"Most frequent RR playing-XI listings, IPL {start_year}–{end_year}"
    return result


def _pct(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(pct=True, ascending=higher_is_better) * 100


def _pressure_metrics(rows: pd.DataFrame, role: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    d = rows.copy()
    if role == "Batter":
        d = d.loc[d["batter_ball_faced"]].copy()
        if d.empty:
            return pd.DataFrame()
        d["player_key"] = _key(d, "batter_id", "batter")
        d["dismissal"] = d["dismissed_striker"].astype(int)
        d["boundary"] = (d["is_boundary_four"] | d["is_boundary_six"]).astype(int)
        result = d.groupby(["player_key", "batter"], as_index=False).agg(
            pressure_deliveries=("batter_ball_faced", "sum"),
            pressure_matches=("match_id", "nunique"),
            pressure_runs=("runs_batter", "sum"),
            dismissals=("dismissal", "sum"), boundaries=("boundary", "sum"),
            mean_pressure_score=("role_pressure_score", "mean"),
        ).rename(columns={"batter": "player_name"})
        result["pressure_strike_rate"] = 100 * result["pressure_runs"] / result["pressure_deliveries"].replace(0, np.nan)
        result["pressure_boundary_rate"] = 100 * result["boundaries"] / result["pressure_deliveries"].replace(0, np.nan)
        result["pressure_dismissals_per_100"] = 100 * result["dismissals"] / result["pressure_deliveries"].replace(0, np.nan)
        result["performance_score"] = (
            .65 * _pct(result["pressure_strike_rate"])
            + .20 * _pct(result["pressure_boundary_rate"])
            + .15 * _pct(result["pressure_dismissals_per_100"], False)
        )
        return result
    if d.empty:
        return pd.DataFrame()
    d["player_key"] = _key(d, "bowler_id", "bowler")
    d["is_legal_metric_ball"] = d["is_legal_ball"].fillna(False).astype(bool)
    d["legal_match_id"] = d["match_id"].where(d["is_legal_metric_ball"])
    # Runs on wides/no-balls count against the bowler even though they do not
    # increase the legal-ball denominator. Dots are legal balls with zero runs.
    d["bowler_runs_conceded"] = d["runs_batter"] + d["wides"] + d["no_balls"]
    delivery_runs = pd.to_numeric(
        d.get("runs_total", d["bowler_runs_conceded"]), errors="coerce"
    ).fillna(0)
    d["dot"] = (delivery_runs.eq(0) & d["is_legal_metric_ball"]).astype(int)
    result = d.groupby(["player_key", "bowler"], as_index=False).agg(
        pressure_deliveries=("is_legal_metric_ball", "sum"),
        pressure_matches=("legal_match_id", "nunique"),
        pressure_runs_conceded=("bowler_runs_conceded", "sum"),
        pressure_wickets=("bowler_credited_wicket", "sum"),
        pressure_dots=("dot", "sum"),
        mean_pressure_score=("role_pressure_score", "mean"),
    ).rename(columns={"bowler": "player_name"})
    result["pressure_economy"] = 6 * result["pressure_runs_conceded"] / result["pressure_deliveries"].replace(0, np.nan)
    result["pressure_wickets_per_100"] = 100 * result["pressure_wickets"] / result["pressure_deliveries"].replace(0, np.nan)
    result["pressure_dot_rate"] = 100 * result["pressure_dots"] / result["pressure_deliveries"].replace(0, np.nan)
    result["performance_score"] = (
        .55 * _pct(result["pressure_economy"], False)
        + .30 * _pct(result["pressure_wickets_per_100"])
        + .15 * _pct(result["pressure_dot_rate"])
    )
    return result


def build_phase_player_rankings(all_ipl: pd.DataFrame) -> pd.DataFrame:
    """Create phase-aware PI and XGBoost pressure rankings for both roles.

    PI selection is adaptive by innings phase: Powerplay=baseline,
    Middle=momentum-heavy, Death=RRR-situation-heavy. For bowlers, the
    selected batting-side PI is inverted to represent defending-side pressure.
    """
    outputs = []
    for role in ["Batter", "Bowler"]:
        for phase in ["Powerplay", "Middle", "Death"]:
            if role == "Batter":
                base = all_ipl.loc[all_ipl["phase"].eq(phase) & all_ipl["batter_ball_faced"]].copy()
                base["player_key"] = _key(base, "batter_id", "batter")
                base["player_name"] = base["batter"]
                specs = [
                    ("PI", "is_phase_pi_high_pressure", "phase_pi_score", "pi"),
                    ("XGBoost", "is_xgb_high_pressure", "xgb_pressure_score", "xgb"),
                ]
            else:
                # Keep illegal deliveries for bowler runs-conceded accounting;
                # the pressure metric denominator is restricted to legal balls.
                base = all_ipl.loc[all_ipl["phase"].eq(phase)].copy()
                base["player_key"] = _key(base, "bowler_id", "bowler")
                base["player_name"] = base["bowler"]
                base["phase_delivery_unit"] = base["is_legal_ball"].fillna(False).astype(int)
                specs = [
                    ("PI", "is_phase_bowling_pi_high_pressure", "phase_bowling_pi_score", "pi"),
                    ("XGBoost", "is_bowling_xgb_high_pressure", "bowling_xgb_pressure_score", "xgb"),
                ]
            if base.empty:
                continue
            canonical_names = base.groupby("player_key")["player_name"].agg(
                lambda names: names.value_counts().index[0]
            )
            base["player_name"] = base["player_key"].map(canonical_names)
            if role == "Batter":
                base["batter"] = base["player_name"]
            else:
                base["bowler"] = base["player_name"]
            if role == "Batter":
                base["phase_delivery_unit"] = 1
            base["phase_match_id"] = base["match_id"].where(base["phase_delivery_unit"].gt(0))
            overall = base.groupby(["player_key", "player_name"], as_index=False).agg(
                phase_deliveries=("phase_delivery_unit", "sum"),
                phase_matches=("phase_match_id", "nunique"),
            )
            for _, flag_col, score_col, prefix in specs:
                subset = base.loc[
                    base["chase_active"].fillna(False)
                    & base[flag_col].fillna(False)
                    & base[score_col].notna()
                ].copy()
                subset["role_pressure_score"] = subset[score_col]
                metrics = _pressure_metrics(subset, role)
                if metrics.empty:
                    continue
                metrics = metrics[[
                    "player_key", "pressure_deliveries", "pressure_matches",
                    "mean_pressure_score", "performance_score",
                ]].rename(columns={
                    "pressure_deliveries": f"{prefix}_pressure_deliveries",
                    "pressure_matches": f"{prefix}_pressure_matches",
                    "mean_pressure_score": f"{prefix}_mean_pressure_score",
                    "performance_score": f"{prefix}_performance_score",
                })
                overall = overall.merge(metrics, on="player_key", how="left")

            for prefix in ["pi", "xgb"]:
                for suffix in ["pressure_deliveries", "pressure_matches", "mean_pressure_score", "performance_score"]:
                    col = f"{prefix}_{suffix}"
                    if col not in overall:
                        overall[col] = np.nan
                overall[f"{prefix}_percentile"] = overall[f"{prefix}_performance_score"]
                overall[f"{prefix}_rank"] = overall[f"{prefix}_performance_score"].rank(method="min", ascending=False)

            pi_ok = overall["pi_percentile"].notna()
            xgb_ok = overall["xgb_percentile"].notna()
            coverage = PI_WEIGHT * pi_ok.astype(float) + XGB_WEIGHT * xgb_ok.astype(float)
            weighted = PI_WEIGHT * overall["pi_percentile"].fillna(0) + XGB_WEIGHT * overall["xgb_percentile"].fillna(0)
            overall["model_weight_coverage"] = coverage
            overall["decision_score"] = (weighted / coverage.replace(0, np.nan)).round(2)
            overall["decision_rank"] = overall["decision_score"].rank(method="min", ascending=False)
            overall["role"] = role
            overall["phase"] = phase
            overall["pi_type"] = PHASE_PI_TYPE[phase]
            overall["decision_method"] = (
                f"80% phase-selected PI ({PHASE_PI_TYPE[phase]}) / 20% synthetic XGBoost; "
                "available model weights renormalised"
            )
            outputs.append(overall)
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["role", "phase", "decision_rank"], na_position="last"
    ).reset_index(drop=True)


def build_frequent_xi_phase_options(frequent_xi: pd.DataFrame,
                                    phase_rankings: pd.DataFrame) -> pd.DataFrame:
    if frequent_xi.empty:
        return pd.DataFrame()
    ids = frequent_xi["player_id"].astype(str).tolist()
    options = pd.MultiIndex.from_product(
        [ids, ["Batter", "Bowler"], ["Powerplay", "Middle", "Death"]],
        names=["player_key", "role", "phase"],
    ).to_frame(index=False)
    xi = frequent_xi.copy()
    xi["player_key"] = xi["player_id"].astype(str)
    options = options.merge(
        xi[["player_key", "historical_xi_rank", "player_name", "appearances", "usage_profile"]],
        on="player_key", how="left",
    )
    if not phase_rankings.empty:
        columns = [
            "player_key", "role", "phase", "phase_deliveries", "phase_matches",
            "pi_pressure_deliveries", "pi_pressure_matches", "pi_performance_score", "pi_rank",
            "xgb_pressure_deliveries", "xgb_pressure_matches", "xgb_performance_score", "xgb_rank",
            "decision_score", "model_weight_coverage", "decision_rank", "pi_type",
        ]
        options = options.merge(
            phase_rankings[[c for c in columns if c in phase_rankings]],
            on=["player_key", "role", "phase"], how="left",
        )
    options["phase_pi_type"] = options.get("pi_type", options["phase"].map(PHASE_PI_TYPE))
    options["phase_option_status"] = np.where(
        options["decision_score"].notna(), "Scored from PI/XG high-pressure samples",
        "No qualifying PI/XG high-pressure sample in this phase",
    )
    return options.sort_values(["historical_xi_rank", "role", "phase"]).reset_index(drop=True)
