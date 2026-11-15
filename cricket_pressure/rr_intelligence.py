from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Iterable
import json
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd

CRICSHEET_IPL_JSON_URL = "https://cricsheet.org/downloads/ipl_json.zip"
CRICSHEET_ATTRIBUTION = (
    "Match data: Cricsheet (https://cricsheet.org/), "
    "Open Data Commons Attribution License (ODC-By)."
)
BOWLER_CREDITED_DISMISSALS = {
    "bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"
}


def download_cricsheet_archive(destination: str | Path, timeout: int = 120) -> Path:
    """Download the official Cricsheet IPL JSON archive to a local ignored path."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        CRICSHEET_IPL_JSON_URL,
        headers={"User-Agent": "Cricket-Aanalysis RR Squad Intelligence/1.0"},
    )
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("Cricsheet download did not return a valid ZIP archive.")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _player_id(info: dict[str, Any], name: str) -> str | None:
    return (info.get("registry") or {}).get("people", {}).get(name)


def _season_year(info: dict[str, Any], match_date: pd.Timestamp) -> int:
    match = re.search(r"(?:19|20)\d{2}", str(info.get("season", "")))
    return int(match.group(0)) if match else int(match_date.year)


def parse_cricsheet_match(payload: dict[str, Any], source_name: str) -> pd.DataFrame:
    """Return one row per delivery, with pre-ball state and role-specific chase pressure."""
    info = payload.get("info") or {}
    teams = info.get("teams") or []
    dates = info.get("dates") or []
    if len(teams) != 2 or not dates:
        raise ValueError(f"{source_name}: incomplete match metadata")
    match_date = pd.to_datetime(dates[0], utc=True, errors="coerce")
    if pd.isna(match_date):
        raise ValueError(f"{source_name}: invalid match date")

    event = info.get("event") or {}
    outcome = info.get("outcome") or {}
    toss = info.get("toss") or {}
    season = _season_year(info, match_date)
    scheduled_overs = max(1, _int(info.get("overs"), 20))
    base = {
        "match_id": Path(source_name).stem,
        "source_file": Path(source_name).name,
        "match_date": match_date,
        "season": season,
        "event_name": event.get("name", ""),
        "venue": info.get("venue", ""),
        "city": info.get("city", ""),
        "team_1": teams[0],
        "team_2": teams[1],
        "winner": outcome.get("winner"),
        "outcome_method": outcome.get("method", ""),
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "player_of_match": "; ".join(info.get("player_of_match") or []),
        "scheduled_overs": scheduled_overs,
    }

    parsed_innings: list[dict[str, Any]] = []
    for innings_number, innings in enumerate(payload.get("innings") or [], start=1):
        if innings.get("super_over"):
            continue
        batting_team = innings.get("team")
        if batting_team not in teams:
            continue
        bowling_team = teams[1] if batting_team == teams[0] else teams[0]
        target_info = innings.get("target") or {}
        innings_overs = max(1, _int(target_info.get("overs"), scheduled_overs))
        parsed: list[dict[str, Any]] = []
        for over_item in innings.get("overs") or []:
            over_number = _int(over_item.get("over"))
            legal_ball_number = 0
            for delivery in over_item.get("deliveries") or []:
                extras = delivery.get("extras") or {}
                runs = delivery.get("runs") or {}
                wickets = delivery.get("wickets") or []
                batter = delivery.get("batter", "Unknown")
                bowler = delivery.get("bowler", "Unknown")
                batter_runs = _int(runs.get("batter"))
                total_runs = _int(runs.get("total"), batter_runs + _int(runs.get("extras")))
                wides = _int(extras.get("wides"))
                no_balls = _int(extras.get("noballs", extras.get("no_balls", 0)))
                wicket_types = [str(w.get("kind", "")).strip().lower() for w in wickets]
                dismissed_players = [w.get("player_out") for w in wickets if w.get("player_out")]
                is_legal = wides == 0 and no_balls == 0
                parsed.append({
                    **base,
                    "innings_number": innings_number,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "over_number": over_number,
                    "ball_in_over": legal_ball_number + 1,
                    "delivery_sequence": len(parsed) + 1,
                    "batter": batter,
                    "batter_id": _player_id(info, batter),
                    "non_striker": delivery.get("non_striker", "Unknown"),
                    "bowler": bowler,
                    "bowler_id": _player_id(info, bowler),
                    "runs_batter": batter_runs,
                    "runs_total": total_runs,
                    "wides": wides,
                    "no_balls": no_balls,
                    "byes": _int(extras.get("byes")),
                    "legbyes": _int(extras.get("legbyes")),
                    "penalty_runs": _int(extras.get("penalty")),
                    "is_legal_ball": is_legal,
                    "batter_ball_faced": wides == 0,
                    "wicket_count": len(wickets),
                    "wicket_types": "; ".join(wicket_types),
                    "dismissed_players": "; ".join(dismissed_players),
                    "dismissed_striker": batter in dismissed_players,
                    "bowler_credited_wicket": any(
                        kind in BOWLER_CREDITED_DISMISSALS for kind in wicket_types
                    ),
                    "is_boundary_four": batter_runs == 4 and not runs.get("non_boundary", False),
                    "is_boundary_six": batter_runs == 6 and not runs.get("non_boundary", False),
                    "innings_scheduled_overs": innings_overs,
                })
                if is_legal:
                    legal_ball_number += 1
        if parsed:
            parsed_innings.append({
                "number": innings_number,
                "target": target_info,
                "deliveries": parsed,
                "total": sum(row["runs_total"] for row in parsed),
            })

    if not parsed_innings:
        return pd.DataFrame()
    first_innings_total = next(
        (inn["total"] for inn in parsed_innings if inn["number"] == 1), np.nan
    )
    second_innings = next((inn for inn in parsed_innings if inn["number"] == 2), None)
    if second_innings and second_innings["target"].get("runs") is not None:
        chase_target = _int(second_innings["target"].get("runs"))
        target_source = "Cricsheet innings target"
    else:
        chase_target = first_innings_total + 1 if pd.notna(first_innings_total) else np.nan
        target_source = "first innings total + 1 (provisional)"
    method = str(base["outcome_method"]).lower()
    dls_without_target = (
        target_source.startswith("first innings")
        and any(token in method for token in ("d/l", "dls", "duckworth"))
    )

    rows: list[dict[str, Any]] = []
    for innings in parsed_innings:
        score = 0
        wickets_before = 0
        legal_balls_before = 0
        recent_legal_runs: deque[int] = deque(maxlen=12)
        for row in innings["deliveries"]:
            total_runs = row["runs_total"]
            scheduled_balls = row["innings_scheduled_overs"] * 6
            balls_remaining = max(0, scheduled_balls - legal_balls_before)
            recent_rate = float(np.mean(recent_legal_runs) * 6) if len(recent_legal_runs) >= 3 else 8.0
            current_rr = score / (legal_balls_before / 6) if legal_balls_before else np.nan
            target = chase_target if row["innings_number"] == 2 else np.nan
            runs_required = max(0, target - score) if pd.notna(target) else np.nan
            rrr = runs_required / (balls_remaining / 6) if row["innings_number"] == 2 and balls_remaining else np.nan
            wickets_remaining = max(0, 10 - wickets_before)
            pressure = np.nan
            bowling_pressure = np.nan
            if row["innings_number"] == 2 and pd.notna(rrr) and not dls_without_target:
                rrr_component = float(np.clip(rrr / 18, 0, 1))
                wicket_component = float(np.clip(1 - wickets_remaining / 10, 0, 1))
                time_component = float(np.clip(1 - balls_remaining / max(scheduled_balls, 1), 0, 1))
                momentum_component = float(np.clip(max(rrr - recent_rate, 0) / 12, 0, 1))
                pressure = 100 * (
                    0.35 * rrr_component + 0.25 * wicket_component
                    + 0.15 * time_component + 0.25 * momentum_component
                )
                scoring_threat = float(np.clip(recent_rate / 15, 0, 1))
                wickets_in_hand = float(np.clip(wickets_remaining / 10, 0, 1))
                chase_control = float(np.clip((recent_rate - rrr + 8) / 16, 0, 1))
                bowling_pressure = 100 * (
                    0.30 * scoring_threat + 0.25 * wickets_in_hand
                    + 0.20 * time_component + 0.25 * chase_control
                )
            phase_end_powerplay = min(36, int(scheduled_balls * 0.30))
            phase_start_death = int(scheduled_balls * 0.80)
            if legal_balls_before < phase_end_powerplay:
                phase = "Powerplay"
            elif legal_balls_before >= phase_start_death:
                phase = "Death"
            else:
                phase = "Middle"

            rows.append({
                **row,
                "score_before": score,
                "wickets_before": wickets_before,
                "legal_balls_before": legal_balls_before,
                "balls_remaining": balls_remaining,
                "wickets_remaining": wickets_remaining,
                "current_run_rate": current_rr,
                "target": target,
                "target_source": target_source if row["innings_number"] == 2 else "",
                "target_is_provisional": bool(dls_without_target and row["innings_number"] == 2),
                "runs_required": runs_required,
                "required_run_rate": rrr,
                "recent_run_rate_12": recent_rate,
                "pressure_index": pressure,
                "is_high_pressure": bool(pd.notna(pressure) and pressure >= 60 and runs_required > 0),
                "bowling_pressure_index": bowling_pressure,
                "is_bowling_high_pressure": bool(
                    pd.notna(bowling_pressure) and bowling_pressure >= 60 and runs_required > 0
                ),
                "phase": phase,
            })
            score += total_runs
            wickets_before += row["wicket_count"]
            if row["is_legal_ball"]:
                recent_legal_runs.append(total_runs)
                legal_balls_before += 1
    return pd.DataFrame(rows)


def _iter_json_payloads(source: str | Path) -> Iterable[tuple[str, dict]]:
    path = Path(source)
    if path.is_dir():
        for file_path in sorted(path.rglob("*.json")):
            yield str(file_path), json.loads(file_path.read_text(encoding="utf-8"))
    elif path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.lower().endswith(".json") and not name.endswith("/"):
                    with archive.open(name) as handle:
                        yield name, json.load(handle)
    else:
        raise FileNotFoundError(f"Expected a Cricsheet JSON folder or ZIP: {path}")


def _role_phase_metrics(frame: pd.DataFrame, team: str, role: str, phase: str) -> dict:
    d = frame.loc[frame["phase"].eq(phase)].copy()
    if role == "Batting":
        team_mask = pd.Series(True, index=d.index) if team == "__OTHER_TEAMS__" else d["batting_team"].eq(team)
        d = d.loc[team_mask & d["batter_ball_faced"]]
        balls = len(d)
        runs = d["runs_batter"].sum()
        dismissals = d["dismissed_striker"].sum()
        boundaries = d["is_boundary_four"].sum() + d["is_boundary_six"].sum()
        return {
            "balls": balls,
            "strike_rate": 100 * runs / balls if balls else np.nan,
            "dismissals_per_100_balls": 100 * dismissals / balls if balls else np.nan,
            "boundary_rate": 100 * boundaries / balls if balls else np.nan,
            "matches": d["match_id"].nunique(),
        }
    team_mask = pd.Series(True, index=d.index) if team == "__OTHER_TEAMS__" else d["bowling_team"].eq(team)
    d = d.loc[team_mask & d["is_legal_ball"]]
    balls = len(d)
    conceded = d["runs_batter"] + d["wides"] + d["no_balls"]
    wickets = d["bowler_credited_wicket"].sum()
    dots = conceded.eq(0).sum()
    return {
        "balls": balls,
        "economy": 6 * conceded.sum() / balls if balls else np.nan,
        "wickets_per_100_balls": 100 * wickets / balls if balls else np.nan,
        "dot_rate": 100 * dots / balls if balls else np.nan,
        "matches": d["match_id"].nunique(),
    }


def build_phase_gaps(rr_deliveries: pd.DataFrame, all_ipl: pd.DataFrame,
                     rr_team: str = "Rajasthan Royals") -> pd.DataFrame:
    rows = []
    for role in ["Batting", "Bowling"]:
        for phase in ["Powerplay", "Middle", "Death"]:
            rr = _role_phase_metrics(rr_deliveries, rr_team, role, phase)
            league = all_ipl.loc[
                (all_ipl["batting_team"].ne(rr_team) if role == "Batting"
                 else all_ipl["bowling_team"].ne(rr_team))
            ]
            baseline = _role_phase_metrics(league, "__OTHER_TEAMS__", role, phase)
            if rr["balls"] == 0 or baseline["balls"] == 0:
                continue
            if role == "Batting":
                need = (
                    0.45 * np.clip((baseline["strike_rate"] - rr["strike_rate"]) / 20, 0, 1)
                    + 0.30 * np.clip((rr["dismissals_per_100_balls"] - baseline["dismissals_per_100_balls"]) / 2, 0, 1)
                    + 0.25 * np.clip((baseline["boundary_rate"] - rr["boundary_rate"]) / 5, 0, 1)
                )
            else:
                need = (
                    0.45 * np.clip((rr["economy"] - baseline["economy"]) / 2, 0, 1)
                    + 0.35 * np.clip((baseline["wickets_per_100_balls"] - rr["wickets_per_100_balls"]) / 2, 0, 1)
                    + 0.20 * np.clip((baseline["dot_rate"] - rr["dot_rate"]) / 5, 0, 1)
                )
            rows.append({
                "role": role, "phase": phase,
                **{f"rr_{k}": v for k, v in rr.items()},
                **{f"league_{k}": v for k, v in baseline.items()},
                "team_need_score": round(float(need * 100), 2),
            })
    return pd.DataFrame(rows)


def build_player_fit_shortlist(all_ipl: pd.DataFrame, phase_gaps: pd.DataFrame) -> pd.DataFrame:
    """Build role/phase percentile shortlists; auction availability is not inferred."""
    outputs = []
    for role in ["Batter", "Bowler"]:
        for phase in ["Powerplay", "Middle", "Death"]:
            d = all_ipl.loc[all_ipl["phase"].eq(phase)].copy()
            if role == "Batter":
                d = d.loc[d["batter_ball_faced"]]
                player_col, name_col = "batter_id", "batter"
                d["metric_runs"] = d["runs_batter"]
                d["metric_wickets"] = d["dismissed_striker"].astype(int)
                d["metric_dots"] = d["runs_batter"].eq(0).astype(int)
                d["metric_boundaries"] = d["is_boundary_four"].astype(int) + d["is_boundary_six"].astype(int)
            else:
                d = d.loc[d["is_legal_ball"]]
                player_col, name_col = "bowler_id", "bowler"
                d["metric_runs"] = d["runs_batter"] + d["wides"] + d["no_balls"]
                d["metric_wickets"] = d["bowler_credited_wicket"].astype(int)
                d["metric_dots"] = d["metric_runs"].eq(0).astype(int)
                d["metric_boundaries"] = d["is_boundary_four"].astype(int) + d["is_boundary_six"].astype(int)
            if d.empty:
                continue
            d["player_id"] = d[player_col].fillna(d[name_col])
            pressure_column = "is_high_pressure" if role == "Batter" else "is_bowling_high_pressure"
            d["pressure_flag"] = d[pressure_column].fillna(False).astype(bool)
            d["franchise"] = d["batting_team"] if role == "Batter" else d["bowling_team"]
            grouped = d.groupby(["player_id", name_col], dropna=False).agg(
                deliveries=("match_id", "size"),
                runs=("metric_runs", "sum"),
                wickets=("metric_wickets", "sum"),
                dots=("metric_dots", "sum"),
                boundaries=("metric_boundaries", "sum"),
                matches=("match_id", "nunique"),
                pressure_deliveries=("pressure_flag", "sum"),
                pressure_runs=("metric_runs", lambda s: s[d.loc[s.index, "pressure_flag"]].sum()),
                pressure_matches=("match_id", lambda s: s[d.loc[s.index, "pressure_flag"]].nunique()),
                franchises=("franchise", lambda s: "; ".join(sorted(set(s.dropna().astype(str))))),
            ).reset_index().rename(columns={name_col: "player_name"})
            grouped["strike_rate_or_economy"] = (
                100 * grouped["runs"] / grouped["deliveries"] if role == "Batter"
                else 6 * grouped["runs"] / grouped["deliveries"]
            )
            grouped["boundary_rate"] = 100 * grouped["boundaries"] / grouped["deliveries"].replace(0, np.nan)
            grouped["dot_rate"] = 100 * grouped["dots"] / grouped["deliveries"].replace(0, np.nan)
            grouped["wicket_rate"] = 100 * grouped["wickets"] / grouped["deliveries"].replace(0, np.nan)
            grouped["pressure_rate"] = (
                100 * grouped["pressure_runs"] / grouped["pressure_deliveries"].replace(0, np.nan)
                if role == "Batter" else
                6 * grouped["pressure_runs"] / grouped["pressure_deliveries"].replace(0, np.nan)
            )
            if role == "Batter":
                sr_pct = grouped["strike_rate_or_economy"].rank(pct=True)
                boundary_pct = grouped["boundary_rate"].rank(pct=True)
                dismissal_pct = grouped["wicket_rate"].rank(pct=True, ascending=False)
                pressure_pct = grouped["pressure_rate"].rank(pct=True).where(
                    grouped["pressure_deliveries"].ge(30), sr_pct
                )
                grouped["player_fit_score"] = 100 * (
                    .40 * sr_pct + .25 * boundary_pct + .20 * dismissal_pct + .15 * pressure_pct
                )
            else:
                economy_pct = grouped["strike_rate_or_economy"].rank(pct=True, ascending=False)
                wicket_pct = grouped["wicket_rate"].rank(pct=True)
                dot_pct = grouped["dot_rate"].rank(pct=True)
                pressure_pct = grouped["pressure_rate"].rank(pct=True, ascending=False).where(
                    grouped["pressure_deliveries"].ge(30), economy_pct
                )
                grouped["player_fit_score"] = 100 * (
                    .35 * economy_pct + .30 * wicket_pct + .20 * dot_pct + .15 * pressure_pct
                )
            need = phase_gaps.loc[
                phase_gaps["role"].eq(role) & phase_gaps["phase"].eq(phase), "team_need_score"
            ]
            need_score = float(need.iloc[0]) if not need.empty else 0.0
            grouped["role"] = role
            grouped["phase"] = phase
            grouped["rr_team_need_score"] = need_score
            grouped["auction_fit_score"] = .55 * grouped["player_fit_score"] + .45 * need_score
            grouped["pressure_sample_adequate"] = grouped["pressure_deliveries"].ge(30)
            grouped["availability_status"] = "Not assessed — requires official auction/retention pool"
            outputs.append(grouped)
    if not outputs:
        return pd.DataFrame()
    result = pd.concat(outputs, ignore_index=True)
    return result.sort_values(
        ["role", "phase", "auction_fit_score", "player_fit_score"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def _match_summary(rr_deliveries: pd.DataFrame, rr_team: str) -> pd.DataFrame:
    columns = ["match_id", "match_date", "season", "venue", "team_1", "team_2",
               "winner", "outcome_method", "toss_winner", "toss_decision", "player_of_match"]
    result = rr_deliveries[columns].drop_duplicates("match_id").copy()
    totals = rr_deliveries.groupby(["match_id", "innings_number"], as_index=False).agg(
        runs=("runs_total", "sum"), wickets=("wicket_count", "sum"),
        legal_balls=("is_legal_ball", "sum"),
    )
    for innings_no in [1, 2]:
        part = totals.loc[totals["innings_number"].eq(innings_no)].set_index("match_id")
        for col in ["runs", "wickets", "legal_balls"]:
            result[f"innings_{innings_no}_{col}"] = result["match_id"].map(part[col])
    result["rr_opponent"] = np.where(result["team_1"].eq(rr_team), result["team_2"], result["team_1"])
    result["rr_result"] = np.select(
        [result["winner"].eq(rr_team), result["winner"].notna()], ["W", "L"], default="NR"
    )
    return result.sort_values("match_date").reset_index(drop=True)


def _pressure_windows(pressure_points: pd.DataFrame) -> pd.DataFrame:
    if pressure_points.empty:
        return pd.DataFrame()
    return pressure_points.groupby(
        ["match_id", "season", "match_date", "batting_team", "bowling_team", "phase"],
        as_index=False,
    ).agg(
        high_pressure_deliveries=("delivery_sequence", "count"),
        max_pressure_index=("pressure_index", "max"),
        mean_required_run_rate=("required_run_rate", "mean"),
        min_wickets_remaining=("wickets_remaining", "min"),
        first_over=("over_number", "min"), last_over=("over_number", "max"),
        batters_involved=("batter", "nunique"), bowlers_involved=("bowler", "nunique"),
    ).sort_values(["max_pressure_index", "high_pressure_deliveries"], ascending=False)


def _write_report(path: Path, matches: pd.DataFrame, rr_deliveries: pd.DataFrame,
                  windows: pd.DataFrame, gaps: pd.DataFrame, shortlist: pd.DataFrame,
                  start_year: int, end_year: int, rr_team: str) -> None:
    lines = [
        "# Rajasthan Royals — Squad Intelligence Report", "",
        f"**Coverage:** IPL seasons {start_year}–{end_year}", f"**Team:** {rr_team}",
        f"**Source:** {CRICSHEET_ATTRIBUTION}", "", "## Dataset coverage", "",
        f"- RR matches: **{len(matches):,}**",
        f"- Deliveries in RR matches: **{len(rr_deliveries):,}**",
        f"- High batter-pressure chase deliveries (PI ≥ 60): **{int(rr_deliveries['is_high_pressure'].sum()):,}**",
        f"- High bowler-pressure chase deliveries (bowling PI ≥ 60): **{int(rr_deliveries['is_bowling_high_pressure'].sum()):,}**",
        "- Player comparisons use the complete selected IPL archive, not only RR matches.",
        "", "## RR phase gaps", "",
    ]
    lines.append(gaps[["role", "phase", "team_need_score"]].to_markdown(index=False)
                 if not gaps.empty else "No comparable phase groups were available.")
    lines += ["", "## Highest-pressure RR match windows", ""]
    lines.append(windows.head(12)[[
        "match_date", "batting_team", "bowling_team", "phase", "high_pressure_deliveries",
        "max_pressure_index", "mean_required_run_rate", "min_wickets_remaining",
    ]].to_markdown(index=False) if not windows.empty else "No chase deliveries crossed PI 60.")
    lines += ["", "## Candidate profiles for RR's weaker phases", ""]
    if gaps.empty or shortlist.empty:
        lines.append("Insufficient data for a player shortlist.")
    else:
        for _, gap in gaps.sort_values("team_need_score", ascending=False).iterrows():
            lines += [f"### {gap['role']} — {gap['phase']} (need score {gap['team_need_score']:.1f}/100)", ""]
            picks = shortlist.loc[
                shortlist["role"].eq(gap["role"]) & shortlist["phase"].eq(gap["phase"])
            ].head(5)
            lines.append(picks[["player_name", "franchises", "auction_fit_score", "player_fit_score",
                                "deliveries", "matches", "pressure_deliveries",
                                "pressure_sample_adequate"]].to_markdown(index=False)
                         if not picks.empty else "No player profiles available.")
            lines.append("")
    lines += [
        "## Interpretation and limitations", "",
        "- Pressure Index v1 is a transparent chase-state heuristic, not a psychological measure or validated win-probability model.",
        "- Only second-innings chase states receive target-derived pressure scores; first-innings deliveries remain in the all-context player baseline.",
        "- DLS/adjusted-target matches without an explicit Cricsheet innings target are not pressure-scored.",
        "- Phases are based on legal-ball position: first 30% Powerplay, middle 50%, final 20% Death.",
        "- Player fit scores are within-role and within-phase percentiles; do not compare batters directly with bowlers.",
        "- Availability, retention, overseas status, price and squad balance are not included. Validate candidates against the official auction pool.",
        "- Pressure sample adequacy is flagged at 30 deliveries; use a stricter threshold before final selection.",
        "- Cricsheet JSON is ODC-By licensed; preserve attribution and licence notices in any public reuse.",
        "", "## Reproduce", "",
        "\x60\x60\x60bash",
        "python scripts/build_rr_intelligence.py --download --start-year 2022 --end-year 2026",
        "\x60\x60\x60", "", CRICSHEET_ATTRIBUTION,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_rr_intelligence(source: str | Path, output_dir: str | Path,
                          start_year: int = 2022, end_year: int = 2026,
                          rr_team: str = "Rajasthan Royals") -> dict[str, Path]:
    """Generate RR-only match outputs and all-IPL candidate-fit rankings."""
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    frames = []
    source_path = Path(source)
    is_ipl_zip = source_path.is_file() and source_path.suffix.lower() == ".zip"
    for source_name, payload in _iter_json_payloads(source):
        info = payload.get("info") or {}
        event_name = str((info.get("event") or {}).get("name", "")).lower()
        teams = info.get("teams") or []
        if not is_ipl_zip and "indian premier league" not in event_name and rr_team not in teams:
            continue
        frame = parse_cricsheet_match(payload, source_name)
        if frame.empty:
            continue
        frame = frame.loc[frame["season"].between(start_year, end_year)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError(f"No IPL match JSON found for seasons {start_year}–{end_year}.")
    all_ipl = pd.concat(frames, ignore_index=True)
    all_ipl["is_rr_match"] = all_ipl["team_1"].eq(rr_team) | all_ipl["team_2"].eq(rr_team)
    rr_ids = set(all_ipl.loc[all_ipl["is_rr_match"], "match_id"])
    if not rr_ids:
        raise ValueError(f"No {rr_team} matches found in the selected seasons.")
    rr_deliveries = all_ipl.loc[all_ipl["match_id"].isin(rr_ids)].copy()
    rr_matches = _match_summary(rr_deliveries, rr_team)
    pressure_points = rr_deliveries.loc[rr_deliveries["is_high_pressure"]].copy()
    windows = _pressure_windows(pressure_points)
    gaps = build_phase_gaps(rr_deliveries, all_ipl, rr_team)
    shortlist = build_player_fit_shortlist(all_ipl, gaps)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_ipl_deliveries": out / "all_ipl_deliveries.csv.gz",
        "rr_deliveries": out / "rr_deliveries.csv.gz",
        "rr_matches": out / "rr_match_summary.csv",
        "rr_pressure_points": out / "rr_pressure_points.csv",
        "rr_bowling_pressure_points": out / "rr_bowling_pressure_points.csv",
        "rr_pressure_windows": out / "rr_pressure_windows.csv",
        "rr_phase_gaps": out / "rr_phase_gaps.csv",
        "player_fit_shortlist": out / "rr_player_fit_shortlist.csv",
        "report": out / "rr_intelligence_report.md",
    }
    all_ipl.to_csv(paths["all_ipl_deliveries"], index=False, compression="gzip")
    rr_deliveries.to_csv(paths["rr_deliveries"], index=False, compression="gzip")
    rr_matches.to_csv(paths["rr_matches"], index=False)
    pressure_points.to_csv(paths["rr_pressure_points"], index=False)
    rr_deliveries.loc[rr_deliveries["is_bowling_high_pressure"]].to_csv(
        paths["rr_bowling_pressure_points"], index=False
    )
    windows.to_csv(paths["rr_pressure_windows"], index=False)
    gaps.to_csv(paths["rr_phase_gaps"], index=False)
    shortlist.to_csv(paths["player_fit_shortlist"], index=False)
    _write_report(paths["report"], rr_matches, rr_deliveries, windows, gaps, shortlist,
                  start_year, end_year, rr_team)
    print(f"IPL matches: {all_ipl['match_id'].nunique():,} | deliveries: {len(all_ipl):,}")
    print(f"RR matches: {len(rr_matches):,} | RR deliveries: {len(rr_deliveries):,}")
    print(f"RR high-pressure chase deliveries: {len(pressure_points):,}")
    print(f"Outputs: {out.resolve()}")
    return paths
