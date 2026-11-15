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
