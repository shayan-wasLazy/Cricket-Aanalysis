import pandas as pd

from cricket_pressure.player_rankings import (
    DEFAULT_RRF_WEIGHTS,
    build_weighted_rrf_ensemble,
)


def test_rrf_joins_role_and_model_metrics_without_column_collisions():
    rows = []
    methods = list(DEFAULT_RRF_WEIGHTS)
    for role in ("Batter", "Bowler"):
        for method_index, method in enumerate(methods):
            for player_index, player_name in enumerate(("Player A", "Player B")):
                row = {
                    "role": role,
                    "scoring_method": method,
                    "half_life": "365d",
                    "player_id": 100 + player_index,
                    "player_name": player_name,
                    "pressure_deliveries": 140 - player_index * 10,
                    "pressure_matches": 12 - player_index,
                    "shrunk_per_100": 20.0 - player_index,
                    "selection_score": 95.0 - 10 * player_index - method_index,
                    "pressure_uplift_per_100": 3.0 - player_index,
                    "days_since_last_pressure_match": 30,
                    "last_pressure_match_date": pd.Timestamp("2026-10-01", tz="UTC"),
                }
                if role == "Bowler":
                    row["economy_per_over"] = 7.2 + player_index
                    row["dot_rate"] = 42.0 - player_index
                rows.append(row)

    result = build_weighted_rrf_ensemble(
        pd.DataFrame(rows),
        min_deliveries=50,
        min_matches=5,
        half_life="365d",
        max_days_since_last_match=365,
        min_model_support=3,
    )

    assert len(result.loc[result["role"].eq("Batter")]) == 2
    assert len(result.loc[result["role"].eq("Bowler")]) == 2
    assert "rank_pi_baseline" in result.columns
    assert "selection_score_pi_baseline" in result.columns
    assert "shrunk_per_100_pi_baseline" in result.columns
    assert "pressure_uplift_per_100_xgboost" in result.columns
    assert "economy_per_over_pi_baseline" in result.columns
    assert "dot_rate_xgboost" in result.columns
    assert not result.duplicated(["role", "player_id"]).any()
