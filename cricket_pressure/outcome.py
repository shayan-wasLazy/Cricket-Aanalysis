from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)
from xgboost import XGBClassifier

HISTORICAL_FEATURES = [
    "runs_required",
    "required_run_rate",
    "wickets_remaining",
    "balls_remaining",
    "scheduled_balls",
    "score_before",
    "legal_balls_before",
    "current_run_rate",
    "recent_run_rate_12",
    "score_momentum_gap",
    "target_percentile",
    "match_stakes",
    "is_t20i",
]


def chronological_match_split(
    frame: pd.DataFrame,
    match_col: str = "match_id",
    date_col: str = "match_date",
    train_fraction: float = 0.80,
) -> tuple[pd.Index, pd.Index, pd.Timestamp]:
    """Split whole matches chronologically. No match can cross train/test."""
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    dates = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
    match_dates = (
        pd.DataFrame({match_col: frame[match_col].values, "_date": dates.values})
        .dropna(subset=["_date"])
        .groupby(match_col)["_date"].min()
        .sort_values()
    )
    if len(match_dates) < 20:
        raise ValueError(f"Need at least 20 dated matches; found {len(match_dates)}")
    split_at = min(max(int(np.floor(len(match_dates) * train_fraction)), 1), len(match_dates) - 1)
    cutoff = pd.Timestamp(match_dates.iloc[split_at])
    train_ids = match_dates.index[match_dates < cutoff]
    test_ids = match_dates.index[match_dates >= cutoff]
    if len(train_ids) == 0 or len(test_ids) == 0:
        raise ValueError("Chronological split produced an empty match partition")
    return train_ids, test_ids, cutoff


def _match_balanced_weights(frame: pd.DataFrame, match_col: str = "match_id") -> np.ndarray:
    counts = frame.groupby(match_col)[match_col].transform("size").to_numpy(dtype=float)
    return 1.0 / np.maximum(counts, 1.0)


def _calibration_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: np.ndarray,
    match_ids: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    d = pd.DataFrame({
        "actual_win": y_true.astype(float),
        "win_probability": probabilities.astype(float),
        "sample_weight": sample_weight.astype(float),
        "match_id": match_ids,
    })
    unique = int(d["win_probability"].nunique())
    if unique < 2:
        d["probability_bin"] = "all"
    else:
        d["probability_bin"] = pd.qcut(
            d["win_probability"], q=min(n_bins, unique),
            duplicates="drop", precision=2,
        ).astype(str)
    rows = []
    for label, g in d.groupby("probability_bin", sort=True, dropna=False):
        w = g["sample_weight"].to_numpy()
        rows.append({
            "probability_bin": str(label),
            "mean_predicted_probability": float(np.average(g["win_probability"], weights=w)),
            "observed_win_rate": float(np.average(g["actual_win"], weights=w)),
            "deliveries": int(len(g)),
            "matches": int(g["match_id"].nunique()),
            "weight_sum": float(w.sum()),
        })
    return pd.DataFrame(rows)


def train_historical_win_probability(
    states: pd.DataFrame,
    random_state: int = 42,
    train_fraction: float = 0.80,
    recency_half_life_days: int | None = None,
) -> tuple[XGBClassifier, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a separate model to actual chase outcomes with chronological match holdout.

    The target is whether the batting side in the second innings eventually won.
    Features are pre-delivery game state only. Rows are weighted inversely by
    deliveries per match so long/short matches do not dominate solely by row count.
    """
    required = {
        "match_id", "match_date", "innings", "chase_active", "batting_team_id",
        "win_team_id", *HISTORICAL_FEATURES,
    }
    missing = sorted(required - set(states.columns))
    if missing:
        raise ValueError(f"Historical outcome model is missing columns: {missing}")

    d = states.loc[
        states["innings"].eq(2)
        & states["chase_active"].fillna(False)
        & states["batting_team_id"].notna()
        & states["win_team_id"].notna()
        & pd.to_numeric(states["batting_team_id"], errors="coerce").gt(0)
        & pd.to_numeric(states["win_team_id"], errors="coerce").gt(0)
    ].copy()
    d["batting_team_won"] = (
        pd.to_numeric(d["batting_team_id"], errors="coerce")
        == pd.to_numeric(d["win_team_id"], errors="coerce")
    ).astype(int)
    d = d.loc[d["batting_team_id"].notna() & d["win_team_id"].notna()].copy()
    d["match_date"] = pd.to_datetime(d["match_date"], errors="coerce", utc=True)
    d = d.dropna(subset=["match_date", "match_id"])
    if d["batting_team_won"].nunique() != 2:
        raise ValueError("Historical training sample must contain both batting-team wins and losses")

    train_ids, test_ids, cutoff = chronological_match_split(
        d, train_fraction=train_fraction
    )
    train = d.loc[d["match_id"].isin(train_ids)].copy()
    test = d.loc[d["match_id"].isin(test_ids)].copy()
    if train["batting_team_won"].nunique() != 2 or test["batting_team_won"].nunique() != 2:
        raise ValueError("Both chronological partitions must contain wins and losses")

    X_train = train[HISTORICAL_FEATURES].apply(pd.to_numeric, errors="coerce")
    X_test = test[HISTORICAL_FEATURES].apply(pd.to_numeric, errors="coerce")
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    y_train = train["batting_team_won"].astype(int)
    y_test = test["batting_team_won"].astype(int)
    w_train_match = _match_balanced_weights(train)
    w_test = _match_balanced_weights(test)

    # Recency weights are computed from training dates only. The latest training
    # date is the anchor; the future holdout is never used to set training weights.
    recency_reference_date = train["match_date"].max()
    if recency_half_life_days is None:
        recency_train = np.ones(len(train), dtype=float)
        half_life_label = "unweighted"
    else:
        if recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive or None")
        train_age_days = (
            recency_reference_date - train["match_date"]
        ).dt.total_seconds().to_numpy() / 86400.0
        recency_train = np.power(
            2.0, -np.clip(train_age_days, 0, None) / float(recency_half_life_days)
        )
        half_life_label = f"{recency_half_life_days}d"

    w_train = w_train_match * recency_train
    # Normalize the mean train weight to one to keep XGBoost regularization scale comparable.
    w_train = w_train / max(float(np.mean(w_train)), 1e-12)

    model = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.90,
        min_child_weight=5,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
    probabilities = model.predict_proba(X_test)[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    prior = float(np.average(y_train, weights=w_train))

    metrics = {
        "target": "batting_team_won",
        "target_definition": "Second-innings batting team eventually won the match.",
        "split_method": "chronological by match_date; all deliveries from each match stay in one partition",
        "train_fraction_by_matches": float(train_fraction),
        "recency_half_life": half_life_label,
        "recency_reference_date_utc": recency_reference_date.isoformat(),
        "mean_train_recency_weight": float(np.mean(recency_train)),
        "min_train_recency_weight": float(np.min(recency_train)),
        "max_train_recency_weight": float(np.max(recency_train)),
        "cutoff_date_utc": cutoff.isoformat(),
        "train_matches": int(train["match_id"].nunique()),
        "test_matches": int(test["match_id"].nunique()),
        "train_deliveries": int(len(train)),
        "test_deliveries": int(len(test)),
        "train_win_rate_match_and_recency_weighted": prior,
        "test_win_rate_match_weighted": float(np.average(y_test, weights=w_test)),
        "accuracy_match_weighted": float(accuracy_score(y_test, predicted, sample_weight=w_test)),
        "brier_score_match_weighted": float(brier_score_loss(y_test, probabilities, sample_weight=w_test)),
        "baseline_brier_score_match_weighted": float(
            brier_score_loss(y_test, np.full(len(y_test), prior), sample_weight=w_test)
        ),
        "log_loss_match_weighted": float(log_loss(y_test, probabilities, labels=[0, 1], sample_weight=w_test)),
        "roc_auc_match_weighted": float(roc_auc_score(y_test, probabilities, sample_weight=w_test)),
        "feature_columns": HISTORICAL_FEATURES,
        "interpretation": (
            "This is a separate historical match-outcome model. Its estimated win probability is an empirical "
            "context benchmark, not a direct psychological-pressure label. Performance is evaluated only on "
            "later matches than training data, with match-balanced sample weights."
        ),
    }

    predictions = test[[
        "match_id", "match_date", "comp_name", "match_format", "season",
        "batting_team_id", "win_team_id", "result", "innings_ball_num",
        "score_before", "runs_required", "required_run_rate",
        "wickets_remaining", "balls_remaining", "pressure_index",
    ]].copy()
    predictions["actual_batting_team_won"] = y_test.to_numpy()
    predictions["win_probability"] = probabilities
    predictions["predicted_batting_team_won"] = predicted
    predictions["match_sample_weight"] = w_test
    predictions["training_recency_half_life"] = half_life_label

    calibration = _calibration_table(y_test.to_numpy(), probabilities, w_test, test["match_id"].to_numpy())
    importance = pd.DataFrame({
        "feature": HISTORICAL_FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return model, metrics, predictions, calibration, importance
