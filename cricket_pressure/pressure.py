from __future__ import annotations
import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "rrr_pressure": 0.25,
    "wicket_pressure": 0.20,
    "time_urgency": 0.15,
    "momentum_pressure": 0.15,
    "death_over_pressure": 0.10,
    "target_difficulty": 0.075,
    "match_stakes": 0.075,
}


def add_pressure_features(states: pd.DataFrame, rrr_reference: float = 12.0) -> pd.DataFrame:
    df = states.copy()
    df["rrr_pressure"] = (pd.to_numeric(df["required_run_rate"], errors="coerce") / rrr_reference).clip(0, 1)
    df["wicket_pressure"] = (1 - pd.to_numeric(df["wickets_remaining"], errors="coerce") / 10).clip(0, 1)
    df["time_urgency"] = (
        1 - pd.to_numeric(df["balls_remaining"], errors="coerce") /
        pd.to_numeric(df["scheduled_balls"], errors="coerce").replace(0, np.nan)
    ).clip(0, 1)
    df["momentum_pressure"] = (
        pd.to_numeric(df["score_momentum_gap"], errors="coerce") / rrr_reference
    ).clip(0, 1)
    df["death_over_pressure"] = (
        1 - pd.to_numeric(df["balls_remaining"], errors="coerce") / 30
    ).clip(0, 1)
    df["target_difficulty"] = pd.to_numeric(df["target_percentile"], errors="coerce").clip(0, 1)
    df["match_stakes"] = pd.to_numeric(df["match_stakes"], errors="coerce").clip(0, 1)
    df["recent_run_rate_12"] = pd.to_numeric(df["recent_run_rate_12"], errors="coerce")
    df["pressure_eligible"] = (
        df["chase_active"].fillna(False) & df[list(DEFAULT_WEIGHTS)].notna().all(axis=1)
    )
    return df


def compute_pressure_index(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    output_col: str = "pressure_index",
) -> pd.DataFrame:
    out = df.copy()
    weights = weights or DEFAULT_WEIGHTS
    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError(f"Weights must contain exactly: {list(DEFAULT_WEIGHTS)}")
    total = float(sum(weights.values()))
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")
    out[output_col] = sum(out[k].fillna(0) * w for k, w in weights.items()) * 100
    out.loc[~out["pressure_eligible"], output_col] = np.nan
    out[output_col.replace("index", "category")] = pd.cut(
        out[output_col], bins=[-0.001, 30, 60, 80, 100],
        labels=["Low", "Moderate", "High", "Extreme"], include_lowest=True,
    )
    return out