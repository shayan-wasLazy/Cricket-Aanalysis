import numpy as np
import pandas as pd

from cricket_pressure.rr_final_decisions import _add_phase_pi, _player_metrics


def test_phase_specific_pi_and_bowler_pressure_are_role_correct():
    rows = pd.DataFrame({
        "phase": ["Powerplay", "Middle", "Death", "Powerplay"],
        "chase_active": [True, True, True, False],
        "rrr_pressure": [0.8] * 4,
        "wicket_pressure": [0.7] * 4,
        "time_urgency": [0.6] * 4,
        "momentum_pressure": [0.3] * 4,
        "death_over_pressure": [0.9] * 4,
        "target_difficulty": [0.5] * 4,
        "match_stakes": [0.4] * 4,
        "xgb_pressure_score": [70.0] * 4,
    })

    scored = _add_phase_pi(rows)

    assert np.isclose(scored.loc[0, "phase_pi_score"], 63.25)
    assert np.isclose(scored.loc[1, "phase_pi_score"], 58.5)
    assert np.isclose(scored.loc[2, "phase_pi_score"], 66.7)
    assert np.isclose(scored.loc[0, "phase_bowling_pi_score"], 36.75)
    assert np.isclose(scored.loc[2, "phase_bowling_pi_score"], 33.3)
    assert scored.loc[0, "phase_pi_type"] == "baseline"
    assert scored.loc[1, "phase_pi_type"] == "momentum_heavy"
    assert scored.loc[2, "phase_pi_type"] == "rrr_situation_heavy"
    assert pd.isna(scored.loc[3, "phase_pi_score"])


def test_bowler_pressure_metrics_use_legal_balls_and_credited_wickets():
    deliveries = pd.DataFrame({
        "match_id": [1, 1, 1, 2, 2],
        "bowler_id": ["B"] * 5,
        "bowler": ["Bowler B"] * 5,
        "is_legal_ball": [True, True, False, True, True],
        "runs_batter": [0, 1, 0, 4, 5],
        "wides": [0, 0, 1, 0, 0],
        "no_balls": [0, 0, 0, 0, 0],
        "bowler_credited_wicket": [False, True, False, True, False],
        "phase_pressure": [60, 60, 60, 60, 60],
    })

    result = _player_metrics(deliveries, "Bowler", "phase_pressure").iloc[0]

    assert result["deliveries"] == 4
    assert result["matches"] == 2
    assert result["runs_conceded"] == 10
    assert result["wickets"] == 2
    assert np.isclose(result["economy"], 15.0)
    assert np.isclose(result["wickets_per_100"], 50.0)
    assert np.isclose(result["dot_rate"], 25.0)



def test_last_over_comparison_uses_same_player_low_pressure_baseline():
    from cricket_pressure.rr_final_decisions import _last_over_comparison

    rows = []
    for match_id, flag in [(1, True), (2, True), (3, False), (4, False)]:
        for ball in range(8):
            rows.append({
                "match_id": match_id,
                "batter_id": "BAT-1",
                "batter": "Finisher",
                "batter_ball_faced": True,
                "chase_active": True,
                "balls_remaining": 6 - (ball % 6),
                "phase_pi_high": flag,
                "phase_pi_baseline": not flag,
                "runs_batter": 2 if flag else 1,
                "dismissed_striker": False,
                "is_boundary_four": flag and ball in {1, 3},
                "is_boundary_six": False,
                "phase_pi_score": 75.0 if flag else 20.0,
            })
    metrics = _last_over_comparison(
        pd.DataFrame(rows),
        "Batter",
        "phase_pi_high",
        "phase_pi_baseline",
        "phase_pi_score",
        "last_over",
    )
    finisher = metrics.iloc[0]
    assert finisher["last_over_deliveries"] == 16
    assert finisher["last_over_matches"] == 2
    assert finisher["last_over_sample_adequate"]
    assert np.isclose(finisher["last_over_strike_rate"], 200.0)
    assert np.isclose(finisher["last_over_baseline_strike_rate"], 100.0)
    assert np.isclose(finisher["last_over_uplift"], 100.0)
