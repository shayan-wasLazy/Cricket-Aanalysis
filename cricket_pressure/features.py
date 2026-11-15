from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

SOURCE_FILES = {
    "ball": "ball_by_ball.csv.gz",
    "matches": "matches.csv",
    "competitions": "competitions.csv",
    "teams": "teams.csv",
    "playing_eleven": "playing_eleven.csv",
    "player_mapping": "competition_team_player_mapping.csv",
}


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a source CSV, accepting either plain CSV or gzip-compressed CSV."""
    candidates = [path]
    if path.suffix == ".gz":
        candidates.append(path.with_suffix(""))  # e.g. ball_by_ball.csv
    elif path.suffix == ".csv":
        candidates.append(path.with_suffix(path.suffix + ".gz"))  # e.g. matches.csv.gz

    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate, low_memory=False)

    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Missing required data file. Tried: {tried}")


def load_source_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the six supplied source tables. The ball file may be gzip-compressed."""
    root = Path(data_dir)
    tables = {key: _read_csv(root / name) for key, name in SOURCE_FILES.items()}
    return tables


def _stage_importance(stage: object, knockout: object) -> float:
    s = str(stage).strip().lower()
    score = 0.25  # ordinary league/group fixture baseline
    if "final" in s:
        score = 1.0
    elif "semi" in s or "qualifier" in s or "eliminator" in s:
        score = 0.90
    elif "super 4" in s or "super four" in s:
        score = 0.55
    elif "other" in s:
        score = 0.35
    try:
        if float(knockout) == 1:
            score = max(score, 0.90)
    except (TypeError, ValueError):
        pass
    return float(np.clip(score, 0, 1))


def build_delivery_states(
    tables: dict[str, pd.DataFrame],
    formats: tuple[str, ...] = ("T20I", "T20"),
    default_overs: int = 20,
) -> pd.DataFrame:
    """Build pre-delivery match states for IPL/T20I-style innings.

    Important source-data caveat:
    extras1_runs is a categorical code (wide=2, legbyes=3, byes=4, etc.),
    not a numeric run count. The available CSV does not unambiguously recover
    bye/legbye run amounts. Scores below are therefore provisional and carry
    an explicit unresolved-extra flag.
    """
    ball = tables["ball"].copy()
    matches = tables["matches"].copy()
    competitions = tables["competitions"].copy()

    required_ball = {"match_id", "innings", "innings_ball_num", "runs", "is_wicket",
                     "extras_type", "extras_2_runs", "striker_id", "striker_name",
                     "bowler_id", "bowler_name", "batting_team_id"}
    required_matches = {"match_id", "comp_id", "stage", "knock_out", "win_team_id",
                        "first_inn_revised_overs", "second_inn_revised_overs", "revised_target"}
    required_comp = {"comp_id", "comp_name", "match_format", "season"}
    for label, frame, required in [
        ("ball_by_ball", ball, required_ball), ("matches", matches, required_matches),
        ("competitions", competitions, required_comp),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing required columns: {sorted(missing)}")

    comp_cols = ["comp_id", "comp_name", "match_format", "season"]
    match_context = matches.merge(
        competitions[comp_cols], on="comp_id", how="left", validate="many_to_one"
    )
    match_context["match_format"] = match_context["match_format"].astype(str).str.upper()
    comp_name_ipl = match_context["comp_name"].fillna("").str.lower().str.contains("ipl", regex=False)
    selected = match_context.loc[
        match_context["match_format"].isin(formats) & (comp_name_ipl | match_context["match_format"].eq("T20I"))
    ].copy()
    selected = selected.drop_duplicates("match_id")
    if selected.empty:
        raise ValueError("No T20I/IPL-style matches found. Check competition metadata.")

    context_cols = [
        "match_id", "comp_id", "comp_name", "match_format", "season", "match_date", "stage",
        "home_team_id", "home_team", "away_team_id", "away_team", "win_team_id", "result",
        "first_inn_revised_overs", "second_inn_revised_overs", "revised_target", "knock_out",
    ]
    context_cols = [c for c in context_cols if c in selected.columns]
    bb = ball.loc[ball["match_id"].isin(selected["match_id"])].copy()
    bb = bb.merge(selected[context_cols], on="match_id", how="left", validate="many_to_one")
    bb = bb.loc[bb["innings"].isin([1, 2])].copy()

    extras = bb["extras_type"].fillna("").astype(str).str.lower().str.strip()
    bb["is_wide"] = extras.eq("wide")
    bb["is_no_ball"] = extras.eq("overstep")
    bb["is_legal_ball"] = ~(bb["is_wide"] | bb["is_no_ball"])
    bb["unresolved_bye_legbye"] = extras.isin(["byes", "legbyes"])
    bb["recorded_batter_runs"] = pd.to_numeric(bb["runs"], errors="coerce").fillna(0)
    bb["recorded_additional_runs"] = pd.to_numeric(bb["extras_2_runs"], errors="coerce").fillna(0)
    bb["provisional_total_runs"] = (
        bb["recorded_batter_runs"] + bb["recorded_additional_runs"] +
        (bb["is_wide"] | bb["is_no_ball"]).astype(int)
    )
    if bb["unresolved_bye_legbye"].any():
        warnings.warn(
            f"{int(bb['unresolved_bye_legbye'].sum()):,} bye/legbye deliveries have unresolved run values. "
            "Treat reconstructed scores and downstream PI/XGBoost comparisons as provisional until scorecards are reconciled.",
            RuntimeWarning,
        )

    bb["is_wicket_num"] = bb["is_wicket"].fillna(False).astype(bool).astype(int)
    bb = bb.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    grp = ["match_id", "innings"]
    bb["score_before"] = bb.groupby(grp)["provisional_total_runs"].transform(
        lambda s: s.cumsum().shift(1).fillna(0)
    )
    bb["wickets_before"] = bb.groupby(grp)["is_wicket_num"].transform(
        lambda s: s.cumsum().shift(1).fillna(0)
    )
    bb["legal_balls_before"] = bb.groupby(grp)["is_legal_ball"].transform(
        lambda s: s.astype(int).cumsum().shift(1).fillna(0)
    )

    bb["scheduled_overs"] = float(default_overs)
    first_revised = bb["innings"].eq(1) & bb["first_inn_revised_overs"].notna()
    second_revised = bb["innings"].eq(2) & bb["second_inn_revised_overs"].notna()
    bb.loc[first_revised, "scheduled_overs"] = pd.to_numeric(
        bb.loc[first_revised, "first_inn_revised_overs"], errors="coerce"
    )
    bb.loc[second_revised, "scheduled_overs"] = pd.to_numeric(
        bb.loc[second_revised, "second_inn_revised_overs"], errors="coerce"
    )
    bb["scheduled_overs"] = bb["scheduled_overs"].fillna(default_overs).clip(lower=1)
    bb["scheduled_balls"] = (bb["scheduled_overs"] * 6).round().astype(int)
    bb["balls_remaining"] = (bb["scheduled_balls"] - bb["legal_balls_before"]).clip(lower=0)
    bb["wickets_remaining"] = (10 - bb["wickets_before"]).clip(lower=0, upper=10)
    bb["current_run_rate"] = np.where(
        bb["legal_balls_before"] > 0,
        bb["score_before"] / (bb["legal_balls_before"] / 6), np.nan,
    )

    innings_totals = bb.groupby(["match_id", "innings"], as_index=False).agg(
        provisional_innings_total=("provisional_total_runs", "sum"),
        legal_balls_total=("is_legal_ball", "sum"),
        unresolved_extra_deliveries=("unresolved_bye_legbye", "sum"),
        wickets_total=("is_wicket_num", "sum"),
        scheduled_balls_for_audit=("scheduled_balls", "max"),
    )
    bb = bb.merge(
        innings_totals, on=["match_id", "innings"], how="left", validate="many_to_one"
    )
    first_totals = innings_totals.loc[innings_totals["innings"].eq(1),
        ["match_id", "provisional_innings_total"]].rename(
        columns={"provisional_innings_total": "first_innings_total"}
    )
    first_match_totals = first_totals.merge(
        selected[["match_id", "match_format", "season", "comp_id"]],
        on="match_id", how="left", validate="one_to_one",
    )
    first_innings_rates = innings_totals.loc[innings_totals["innings"].eq(1)].merge(
        selected[["match_id", "match_format", "season"]],
        on="match_id", how="left", validate="one_to_one",
    )
    first_innings_rates["innings_run_rate"] = np.where(
        first_innings_rates["legal_balls_total"].gt(0),
        first_innings_rates["provisional_innings_total"] / (first_innings_rates["legal_balls_total"] / 6),
        np.nan,
    )
    season_rates = first_innings_rates.groupby(["match_format", "season"], dropna=False)["innings_run_rate"].mean().rename("season_baseline_run_rate").reset_index()
    bb = bb.merge(first_totals, on="match_id", how="left", validate="many_to_one")
    bb = bb.merge(season_rates, on=["match_format", "season"], how="left", validate="many_to_one")
    bb["target"] = bb["first_innings_total"] + 1
    revised = bb["revised_target"].notna()
    bb.loc[revised, "target"] = pd.to_numeric(bb.loc[revised, "revised_target"], errors="coerce")
    bb["runs_required"] = (bb["target"] - bb["score_before"]).clip(lower=0)
    bb["required_run_rate"] = np.where(
        bb["innings"].eq(2) & bb["balls_remaining"].gt(0),
        bb["runs_required"] / (bb["balls_remaining"] / 6), np.nan,
    )
    bb["chase_active"] = (
        bb["innings"].eq(2) & bb["runs_required"].gt(0) & bb["balls_remaining"].gt(0)
    )

    # Prior 12 deliveries only; current delivery result cannot leak into its own state.
    bb["prior_12_delivery_runs_mean"] = bb.groupby(grp)["provisional_total_runs"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=3).mean()
    )
    bb["recent_run_rate_12"] = bb["prior_12_delivery_runs_mean"] * 6
    bb["recent_run_rate_12"] = bb["recent_run_rate_12"].fillna(bb["season_baseline_run_rate"])
    bb["score_momentum_gap"] = (bb["required_run_rate"] - bb["recent_run_rate_12"]).clip(lower=0)

    # Competition-season target percentile; this is a relative target-size proxy, not a win probability.
    first_match_totals["target_percentile"] = first_match_totals.groupby(
        ["match_format", "season"]
    )["first_innings_total"].rank(pct=True, method="average")
    bb = bb.merge(
        first_match_totals[["match_id", "target_percentile"]],
        on="match_id", how="left", validate="many_to_one",
    )
    bb["target_percentile"] = bb["target_percentile"].fillna(0.5)
    bb["match_stakes"] = [
        _stage_importance(s, k) for s, k in zip(bb["stage"], bb["knock_out"])
    ]
    bb["is_knockout"] = pd.to_numeric(bb["knock_out"], errors="coerce").fillna(0).astype(int)
    bb["is_t20i"] = bb["match_format"].eq("T20I").astype(int)
    bb["match_id"] = bb["match_id"].astype(int)

    # True mathematically-must-win status cannot be reconstructed from this pack alone.
    bb["must_win_status_available"] = False
    bb["must_win_proxy"] = np.nan
    return bb