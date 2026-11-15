import numpy as np
import pandas as pd

from cricket_pressure.final_selection import (
    CONSENSUS_METHODS,
    build_consensus_rankings,
    build_final_selection_candidates,
)


def _make_pressure_rows():
    rows = []
    # Batter A is consistently good. Batter B has a higher raw mean from a
    # handful of exceptional matches but is much more volatile.
    for match_no in range(10):
        for player_id, name in [(10, "Consistent Batter"), (20, "Volatile Batter")]:
            for ball_no in range(10):
                if player_id == 10:
                    runs = 4
                else:
                    runs = 6 if match_no < 8 else 0
                rows.append({
                    "match_id": match_no + 1,
                    "match_date": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=match_no * 30),
                    "striker_id": player_id,
                    "striker_name": name,
                    "bowler_id": 900 + (match_no % 2),
                    "bowler_name": f"Bowler {900 + (match_no % 2)}",
                    "runs": runs,
                    "is_wide": False,
                    "is_legal_ball": True,
                    "unresolved_bye_legbye": False,
                    "provisional_total_runs": runs,
                    "is_wicket": False,
                    "dismissed_player_id": np.nan,
                    "dismissal_type": "",
                    "pressure_index": 80.0,
                    "pi_rrr_situation_heavy": 80.0,
                    "pi_momentum_heavy": 80.0,
                    "pi_stakes_heavy": 80.0,
                    "xgb_pressure_score": 80.0,
                })
    return pd.DataFrame(rows)


def test_consistency_adjusted_rank_rewards_repeatable_performance():
    candidates = build_final_selection_candidates(
        _make_pressure_rows(), min_deliveries=100, min_matches=10,
        prior_matches=5, half_lives_days=(365,),
    )
    batters = candidates.loc[
        candidates["role"].eq("Batter")
        & candidates["scoring_method"].eq("PI baseline")
        & candidates["half_life"].eq("unweighted")
    ].sort_values("rank")

    assert batters["player_id"].tolist() == [10, 20]
    consistent = batters.iloc[0]
    volatile = batters.iloc[1]
    assert consistent["std_match_rate"] < volatile["std_match_rate"]
    assert consistent["consistency_score"] > volatile["consistency_score"]
    # Volatile Batter's raw average is higher, but inconsistency lowers their score.
    assert volatile["raw_pressure_rate"] > consistent["raw_pressure_rate"]
    assert consistent["pressure_matches"] == 10
    assert consistent["pressure_deliveries"] == 100


def test_consensus_uses_four_pi_variants_and_keeps_xgb_diagnostic_only():
    candidates = build_final_selection_candidates(
        _make_pressure_rows(), min_deliveries=100, min_matches=10,
        half_lives_days=(),
    )
    consensus = build_consensus_rankings(candidates, min_methods=3)
    batters = consensus.loc[
        consensus["role"].eq("Batter") & consensus["half_life"].eq("unweighted")
    ].sort_values("consensus_rank")

    assert batters["player_id"].tolist() == [10, 20]
    assert batters["methods_supported"].eq(4).all()
    assert "XGBoost" in set(candidates["scoring_method"])
    assert set(CONSENSUS_METHODS) == {
        "PI baseline", "PI RRR-heavy", "PI momentum-heavy", "PI stakes-heavy"
    }
    assert set(batters["methods_supported"]) == {4}


def test_sample_gate_requires_pressure_deliveries_and_matches():
    candidates = build_final_selection_candidates(
        _make_pressure_rows(), min_deliveries=101, min_matches=10,
        half_lives_days=(),
    )
    pi = candidates.loc[
        candidates["role"].eq("Batter")
        & candidates["scoring_method"].eq("PI baseline")
    ]
    assert not pi["eligible"].any()


def test_bowler_dot_ball_rate_excludes_unresolved_bye_legbye_rows():
    rows = _make_pressure_rows()
    rows["striker_id"] = 700
    rows["striker_name"] = "Batter"
    rows["runs"] = 1
    rows["provisional_total_runs"] = 1
    rows["bowler_id"] = np.where(rows.index % 2 == 0, 901, 902)
    rows["bowler_name"] = np.where(rows.index % 2 == 0, "Bowler 901", "Bowler 902")
    rows["runs"] = np.where(rows["bowler_id"].eq(901), 0, 1)
    rows["provisional_total_runs"] = rows["runs"]
    rows["unresolved_bye_legbye"] = False

    # Add a row that looks like a dot in the provisional total but is actually
    # an unresolved bye/legbye. It must not enter either side of the dot rate.
    extra = rows.iloc[[0]].copy()
    extra["match_id"] = 999
    extra["bowler_id"] = 901
    extra["bowler_name"] = "Bowler 901"
    extra["unresolved_bye_legbye"] = True
    extra["is_legal_ball"] = True
    extra["provisional_total_runs"] = 0
    rows = pd.concat([rows, extra], ignore_index=True)

    candidates = build_final_selection_candidates(
        rows, min_deliveries=1, min_matches=1, half_lives_days=(),
    )
    bowler = candidates.loc[
        candidates["role"].eq("Bowler")
        & candidates["scoring_method"].eq("PI baseline")
        & candidates["half_life"].eq("unweighted")
        & candidates["player_id"].eq(901)
    ]
    assert len(bowler) == 1
    assert np.isclose(bowler.iloc[0]["raw_pressure_rate"], 100.0)
    assert bowler.iloc[0]["secondary_metric"] == "Credited wickets per 100 legal balls"


def test_all_recency_settings_are_generated():
    candidates = build_final_selection_candidates(_make_pressure_rows())
    assert set(candidates["half_life"].unique()) == {"unweighted", "365d", "540d", "730d"}
