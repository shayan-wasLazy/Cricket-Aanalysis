import pandas as pd

from cricket_pressure.player_rankings import _summarise_role


def test_global_bowler_economy_includes_wides_but_uses_legal_balls_and_matches():
    rows = pd.DataFrame({
        "match_id": [1, 1, 2, 2, 3],
        "match_date": ["2026-10-01", "2026-10-01", "2026-10-02", "2026-10-02", "2026-10-03"],
        "bowler_id": [17] * 5,
        "bowler_name": ["Bowler A"] * 5,
        "is_legal_ball": [True, False, True, True, False],
        "is_wicket": [False, False, True, False, False],
        "dismissal_type": ["", "", "bowled", "", ""],
        "extras_type": ["", "wide", "", "legbyes", "wide"],
        "extras_2_runs": [0, 0, 0, 1, 0],
        "runs": [1, 0, 0, 0, 0],
        "provisional_total_runs": [1, 1, 0, 1, 1],
        "recency_weight": [1.0, 1.0, 1.0, 1.0, 1.0],
    })

    result = _summarise_role(
        rows,
        role="Bowler",
        prior_deliveries=100,
        reference_date=pd.Timestamp("2026-11-16", tz="UTC"),
    ).iloc[0]

    assert result["pressure_deliveries"] == 3
    # Match 3 had only a wide, so it must not count as a legal-ball pressure match.
    assert result["pressure_matches"] == 2
    assert result["raw_runs_conceded"] == 3
    assert result["raw_performance"] == 1
    assert result["economy_per_over"] == 6.0
    assert result["dot_rate"] == (100.0 / 3.0)
