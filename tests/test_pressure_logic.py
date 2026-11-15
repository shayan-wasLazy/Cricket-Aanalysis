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
    assert set(ranks["half_life"].unique()) == {"unweighted","180d","365d","540d","730d"}
    batter = ranks.loc[
        ranks["role"].eq("Batter") & ranks["scoring_method"].eq("PI baseline")
    ]
    assert batter["weighted_per_100"].notna().all()



def test_player_recency_uses_current_selection_date_and_exposes_inactivity():
    rows = pd.DataFrame({
        "match_id": [1, 2],
        "match_date": pd.to_datetime(["2024-11-15", "2026-11-01"], utc=True),
        "striker_id": [10, 20],
        "striker_name": ["Inactive Player", "Active Player"],
        "bowler_id": [30, 40],
        "bowler_name": ["Old Bowler", "Recent Bowler"],
        "runs": [6, 4],
        "is_wicket": [False, False],
        "dismissed_player_id": [np.nan, np.nan],
        "dismissal_type": ["", ""],
        "pressure_index": [75.0, 75.0],
        "xgb_pressure_score": [75.0, 75.0],
    })
    reference = "2026-11-15"
    rankings = build_player_rankings(
        rows,
        min_deliveries=1,
        min_matches=1,
        prior_deliveries=10,
        half_lives_days=(180,),
        reference_date=reference,
    )
    batters = rankings.loc[
        rankings["role"].eq("Batter")
        & rankings["scoring_method"].eq("PI baseline")
        & rankings["half_life"].eq("180d")
    ].set_index("player_name")

    assert batters.loc["Inactive Player", "weighted_deliveries"] < (
        batters.loc["Active Player", "weighted_deliveries"] * 0.1
    )
    assert batters.loc["Inactive Player", "days_since_last_pressure_match"] > 700
    assert batters.loc["Inactive Player", "recent_pressure_matches_365d"] == 0
    assert batters.loc["Active Player", "recent_pressure_matches_365d"] == 1
    assert batters["recency_reference_date"].eq("2026-11-15T00:00:00+00:00").all()


def test_180_day_half_life_reduces_two_year_old_match_to_about_six_percent():
    dates = pd.Series(pd.to_datetime(["2024-11-15", "2026-11-15"], utc=True))
    weights, reference = exponential_recency_weights(
        dates, half_life_days=180, reference_date="2026-11-15"
    )
    age_days = (
        pd.Timestamp("2026-11-15", tz="UTC")
        - pd.Timestamp("2024-11-15", tz="UTC")
    ).total_seconds() / 86400.0
    expected_old_weight = 2 ** (-age_days / 180)
    assert np.isclose(weights.iloc[0], expected_old_weight, atol=1e-9)
    assert np.isclose(weights.iloc[1], 1.0)
    assert str(reference.date()) == "2026-11-15"



def test_rrf_weights_are_deliberately_unequal_and_sum_to_one():
    from cricket_pressure.player_rankings import DEFAULT_RRF_WEIGHTS

    assert np.isclose(sum(DEFAULT_RRF_WEIGHTS.values()), 1.0)
    assert len(set(DEFAULT_RRF_WEIGHTS.values())) == len(DEFAULT_RRF_WEIGHTS)
    assert DEFAULT_RRF_WEIGHTS["PI baseline"] > DEFAULT_RRF_WEIGHTS["XGBoost"]


def test_weighted_rrf_combines_model_ranks_and_requires_model_support():
    from cricket_pressure.player_rankings import (
        DEFAULT_RRF_WEIGHTS,
        build_weighted_rrf_ensemble,
    )

    score_by_method = {
        "PI baseline": {1: 100, 2: 90, 3: 80, 4: 70},
        "PI RRR-heavy": {2: 100, 1: 90, 3: 80, 4: 70},
        "PI momentum-heavy": {3: 100, 2: 90, 1: 80},
        "PI stakes-heavy": {1: 100, 3: 90, 2: 80},
        "XGBoost": {3: 100, 2: 90, 1: 80},
    }
    rows = []
    for method, players in score_by_method.items():
        for player_id, score in players.items():
            rows.append({
                "player_id": player_id,
                "player_name": f"Player {player_id}",
                "role": "Batter",
                "scoring_method": method,
                "half_life": "180d",
                "shrunk_per_100": float(score),
                "pressure_deliveries": 120,
                "pressure_matches": 12,
                "last_pressure_match_date": pd.Timestamp("2026-11-01", tz="UTC"),
                "days_since_last_pressure_match": 10.0,
            })
    rankings = pd.DataFrame(rows)

    ensemble = build_weighted_rrf_ensemble(
        rankings,
        min_deliveries=100,
        min_matches=10,
        half_life="180d",
        max_days_since_last_match=365,
        min_model_support=3,
    )

    # Player 4 appears in only two model lists and is excluded.
    assert 4 not in set(ensemble["player_id"])
    assert ensemble["model_support"].min() >= 3

    # Weighted RRF is deterministic and, for these deliberately constructed
    # rankings, Player 1 narrowly outranks Player 2.
    assert ensemble.iloc[0]["player_id"] == 1
    player_1 = ensemble.set_index("player_id").loc[1]
    expected = (
        DEFAULT_RRF_WEIGHTS["PI baseline"] / (60 + 1)
        + DEFAULT_RRF_WEIGHTS["PI RRR-heavy"] / (60 + 2)
        + DEFAULT_RRF_WEIGHTS["PI momentum-heavy"] / (60 + 3)
        + DEFAULT_RRF_WEIGHTS["PI stakes-heavy"] / (60 + 1)
        + DEFAULT_RRF_WEIGHTS["XGBoost"] / (60 + 3)
    )
    assert np.isclose(player_1["rrf_score"], expected)
    assert player_1["model_support"] == 5
    assert player_1["model_weight_coverage"] == 1.0
    assert player_1["ensemble_rank"] == 1


def test_weighted_rrf_applies_sample_and_activity_gates_per_model():
    from cricket_pressure.player_rankings import build_weighted_rrf_ensemble

    rows = []
    for method in [
        "PI baseline", "PI RRR-heavy", "PI momentum-heavy",
        "PI stakes-heavy", "XGBoost",
    ]:
        rows.extend([
            {
                "player_id": 1, "player_name": "Eligible",
                "role": "Batter", "scoring_method": method, "half_life": "180d",
                "shrunk_per_100": 120.0, "pressure_deliveries": 150,
                "pressure_matches": 15, "last_pressure_match_date": pd.Timestamp("2026-11-01", tz="UTC"),
                "days_since_last_pressure_match": 10.0,
            },
            {
                "player_id": 2, "player_name": "Thin sample",
                "role": "Batter", "scoring_method": method, "half_life": "180d",
                "shrunk_per_100": 300.0, "pressure_deliveries": 20,
                "pressure_matches": 3, "last_pressure_match_date": pd.Timestamp("2026-11-01", tz="UTC"),
                "days_since_last_pressure_match": 10.0,
            },
            {
                "player_id": 3, "player_name": "Inactive",
                "role": "Batter", "scoring_method": method, "half_life": "180d",
                "shrunk_per_100": 250.0, "pressure_deliveries": 150,
                "pressure_matches": 15, "last_pressure_match_date": pd.Timestamp("2024-01-01", tz="UTC"),
                "days_since_last_pressure_match": 1000.0,
            },
        ])

    ensemble = build_weighted_rrf_ensemble(
        pd.DataFrame(rows),
        min_deliveries=100,
        min_matches=10,
        half_life="180d",
        max_days_since_last_match=365,
        min_model_support=3,
    )
    assert ensemble["player_id"].tolist() == [1]
