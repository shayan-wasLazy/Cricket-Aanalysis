import numpy as np
import pandas as pd

from cricket_pressure.final_selection import (
    build_consensus_rankings,
    build_final_selection_candidates,
)


def _selection_fixture():
    rows = []
    for player_id, name, pressure_runs, normal_runs in [
        (10, "Batter A", [4, 4], [2, 2]),
        (20, "Batter B", [2, 2], [4, 4]),
    ]:
        for i in range(2):
            for is_pressure, runs in [
                (True, pressure_runs[i]),
                (False, normal_runs[i]),
            ]:
                row = {
                    "match_id": i + 1,
                    "match_date": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=i * 30),
                    "match_format": "T20",
                    "season": 2025,
                    "striker_id": player_id,
                    "striker_name": name,
                    "bowler_id": player_id + 100,
                    "bowler_name": f"Bowler {name}",
                    "runs": runs,
                    "is_wicket": False,
                    "dismissed_player_id": np.nan,
                    "dismissal_type": "",
                    "is_wide": False,
                    "is_legal_ball": True,
                    "pressure_index": 80.0 if is_pressure else 30.0,
                    "pi_rrr_situation_heavy": 80.0 if is_pressure else 30.0,
                    "pi_momentum_heavy": 80.0 if is_pressure else 30.0,
                    "pi_stakes_heavy": 80.0 if is_pressure else 30.0,
                    # Deliberately reverse the PI grouping to ensure XGBoost is diagnostic only.
                    "xgb_pressure_score": 30.0 if is_pressure else 80.0,
                }
                rows.append(row)
    return pd.DataFrame(rows)


def test_final_selection_compares_pressure_with_same_format_season_baseline():
    candidates = build_final_selection_candidates(
        _selection_fixture(), min_deliveries=2, min_matches=2, prior_balls=10,
        half_lives_days=(365,),
    )
    batters = candidates.loc[
        candidates["role"].eq("Batter")
        & candidates["scoring_method"].eq("PI baseline")
        & candidates["half_life"].eq("unweighted")
    ].sort_values("rank")
    assert batters["player_id"].tolist() == [10, 20]
    assert batters.iloc[0]["uplift_pct"] > 0
    assert batters.iloc[1]["uplift_pct"] < 0
    assert batters.iloc[0]["pressure_deliveries"] == 2
    assert batters.iloc[0]["normal_deliveries"] == 2
    assert batters.iloc[0]["pressure_matches"] == 2
    assert batters.iloc[0]["normal_matches"] == 2


def test_consensus_requires_three_pi_methods_and_excludes_synthetic_xgb():
    candidates = build_final_selection_candidates(
        _selection_fixture(), min_deliveries=2, min_matches=2, prior_balls=10,
        half_lives_days=(365,),
    )
    consensus = build_consensus_rankings(candidates, min_methods=3)
    default = consensus.loc[
        consensus["role"].eq("Batter") & consensus["half_life"].eq("unweighted")
    ].sort_values("consensus_rank")
    assert default["player_id"].tolist() == [10, 20]
    assert default["methods_supported"].eq(4).all()
    assert "XGBoost" in set(candidates["scoring_method"])
    assert default["methods_supported"].eq(4).all()
    assert default.iloc[0]["baseline_pi_uplift_pct"] > default.iloc[1]["baseline_pi_uplift_pct"]


def test_final_selection_eligibility_requires_both_pressure_and_baseline_sample():
    candidates = build_final_selection_candidates(
        _selection_fixture(), min_deliveries=3, min_matches=2,
        half_lives_days=(),
    )
    pi = candidates.loc[
        candidates["role"].eq("Batter")
        & candidates["scoring_method"].eq("PI baseline")
    ]
    assert not pi["eligible"].any()



def test_pressure_and_baseline_are_matched_within_format_and_season():
    rows = _selection_fixture()
    rows["season"] = np.where(rows["pressure_index"].ge(60), 2024, 2025)
    candidates = build_final_selection_candidates(
        rows, min_deliveries=1, min_matches=1, half_lives_days=(),
    )
    assert candidates.empty


def test_bowler_uplift_uses_credited_wickets_per_legal_ball():
    rows = _selection_fixture()
    rows["is_wide"] = False
    rows["is_legal_ball"] = True
    rows["is_wicket"] = False
    rows["dismissal_type"] = ""
    # Give Bowler 110 one credited wicket on each pressure delivery and none in baseline.
    mask = rows["bowler_id"].eq(110) & rows["pressure_index"].ge(60)
    rows.loc[mask, "is_wicket"] = True
    rows.loc[mask, "dismissal_type"] = "caught"
    candidates = build_final_selection_candidates(
        rows, min_deliveries=2, min_matches=2, prior_balls=10,
        half_lives_days=(),
    )
    bowler = candidates.loc[
        candidates["role"].eq("Bowler")
        & candidates["scoring_method"].eq("PI baseline")
        & candidates["half_life"].eq("unweighted")
        & candidates["player_id"].eq(110)
    ]
    assert len(bowler) == 1
    assert bowler.iloc[0]["pressure_rate"] > bowler.iloc[0]["normal_rate"]
    assert bowler.iloc[0]["performance_metric"] == "Credited wickets per 100 legal balls"
