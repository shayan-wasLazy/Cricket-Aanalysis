import numpy as np
import pandas as pd
from cricket_pressure.pressure import DEFAULT_WEIGHTS, add_pressure_features, compute_pressure_index


def _score(required_rate, wickets_remaining, balls_remaining, recent_rate=8.0):
    row = pd.DataFrame([{
        "required_run_rate": required_rate,
        "wickets_remaining": wickets_remaining,
        "balls_remaining": balls_remaining,
        "scheduled_balls": 120,
        "score_momentum_gap": max(required_rate - recent_rate, 0),
        "target_percentile": 0.5,
        "match_stakes": 0.25,
        "recent_run_rate_12": recent_rate,
        "chase_active": True,
    }])
    return compute_pressure_index(add_pressure_features(row))["pressure_index"].iloc[0]


def test_weights_sum_to_one():
    assert np.isclose(sum(DEFAULT_WEIGHTS.values()), 1.0)


def test_higher_required_rate_does_not_reduce_pressure():
    assert _score(14, 8, 60) > _score(8, 8, 60)


def test_fewer_wickets_does_not_reduce_pressure():
    assert _score(10, 2, 60) > _score(10, 9, 60)


def test_fewer_balls_does_not_reduce_pressure():
    assert _score(10, 7, 12) > _score(10, 7, 72)


from cricket_pressure.player_rankings import build_player_rankings
from cricket_pressure.outcome import chronological_match_split


def test_player_rankings_apply_delivery_and_match_eligibility():
    rows = pd.DataFrame({
        "match_id": [1, 2, 1, 1],
        "match_date": pd.to_datetime(["2025-01-01","2025-06-01","2025-01-01","2025-01-01"], utc=True),
        "striker_id": [10, 10, 20, 20],
        "striker_name": ["Batter A", "Batter A", "Batter B", "Batter B"],
        "bowler_id": [30, 30, 40, 40],
        "bowler_name": ["Bowler A", "Bowler A", "Bowler B", "Bowler B"],
        "runs": [4, 2, 6, 0],
        "is_wicket": [False, False, True, False],
        "dismissed_player_id": [np.nan, np.nan, 20, np.nan],
        "dismissal_type": ["", "", "caught", ""],
        "pressure_index": [75.0, 80.0, 70.0, 72.0],
        "xgb_pressure_score": [72.0, 77.0, 68.0, 69.0],
    })
    rankings = build_player_rankings(
        rows, min_deliveries=2, min_matches=2, prior_deliveries=10
    )
    batter_pi = rankings.loc[
        rankings["role"].eq("Batter") & rankings["scoring_method"].eq("PI baseline")
        & rankings["half_life"].eq("unweighted")
    ]
    eligible = batter_pi.loc[batter_pi["eligible"]]
    assert eligible["player_id"].tolist() == [10]
    assert eligible.iloc[0]["pressure_matches"] == 2
    assert eligible.iloc[0]["rank"] == 1


def test_chronological_split_keeps_matches_disjoint():
    frame = pd.DataFrame({
        "match_id": list(range(30)),
        "match_date": pd.date_range("2022-01-01", periods=30, freq="D"),
    })
    train_ids, test_ids, cutoff = chronological_match_split(frame, train_fraction=0.8)
    assert set(train_ids).isdisjoint(set(test_ids))
    assert len(train_ids) == 24
    assert len(test_ids) == 6
    assert frame.loc[frame["match_id"].isin(train_ids), "match_date"].max() < cutoff
    assert frame.loc[frame["match_id"].isin(test_ids), "match_date"].min() >= cutoff


from cricket_pressure.player_rankings import exponential_recency_weights


def test_exponential_recency_weight_has_expected_half_life():
    dates = pd.Series(pd.to_datetime([
        "2025-01-01", "2026-01-01", "2027-01-01"
    ], utc=True))
    weights, reference = exponential_recency_weights(dates, half_life_days=365)
    assert np.isclose(weights.iloc[-1], 1.0)
    assert np.isclose(weights.iloc[-2], 0.5, atol=0.002)
    assert np.isclose(weights.iloc[0], 0.25, atol=0.003)
    assert str(reference.date()) == "2027-01-01"


def test_player_rankings_include_all_recency_settings():
    from cricket_pressure.player_rankings import build_player_rankings

    rows = pd.DataFrame({
        "match_id": [1, 2, 3, 4],
        "match_date": pd.to_datetime(["2025-01-01","2025-06-01","2026-01-01","2026-06-01"], utc=True),
        "striker_id": [10,10,20,20], "striker_name": ["A","A","B","B"],
        "bowler_id": [30,30,40,40], "bowler_name": ["C","C","D","D"],
        "runs": [2,2,6,6], "is_wicket": [False,False,True,True],
        "dismissed_player_id": [np.nan,np.nan,20,20],
        "dismissal_type": ["","","caught","caught"],
        "pressure_index": [75,75,75,75],
        "xgb_pressure_score": [75,75,75,75],
    })
    ranks = build_player_rankings(
        rows, min_deliveries=1, min_matches=1, prior_deliveries=1
    )
    assert set(ranks["half_life"].unique()) == {"unweighted","365d","540d","730d"}
    batter = ranks.loc[
        ranks["role"].eq("Batter") & ranks["scoring_method"].eq("PI baseline")
    ]
    assert batter["weighted_per_100"].notna().all()
