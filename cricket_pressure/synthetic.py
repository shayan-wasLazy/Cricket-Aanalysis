from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, mean_absolute_error, r2_score
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "required_run_rate", "wickets_remaining", "balls_remaining", "scheduled_balls",
    "recent_run_rate_12", "target_difficulty", "match_stakes", "death_over_pressure", "is_t20i",
]
CLASS_NAMES = ["Low", "Moderate", "High", "Extreme"]
CLASS_MIDPOINTS = np.array([15.0, 45.0, 70.0, 90.0])


def _components(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["rrr_pressure"] = (x["required_run_rate"] / 12.0).clip(0, 1)
    x["wicket_pressure"] = (1 - x["wickets_remaining"] / 10.0).clip(0, 1)
    x["time_urgency"] = (1 - x["balls_remaining"] / x["scheduled_balls"].replace(0, np.nan)).clip(0, 1)
    x["momentum_pressure"] = ((x["required_run_rate"] - x["recent_run_rate_12"]).clip(lower=0) / 12.0).clip(0, 1)
    x["death_over_pressure"] = (1 - x["balls_remaining"] / 30.0).clip(0, 1)
    x["target_difficulty"] = x["target_difficulty"].clip(0, 1)
    x["match_stakes"] = x["match_stakes"].clip(0, 1)
    return x


def synthetic_teacher_score(df: pd.DataFrame, seed: int = 11) -> pd.Series:
    """Transparent synthetic scenario rule. This is a test oracle, not real pressure truth."""
    x = _components(df)
    r, w, t, m = x["rrr_pressure"], x["wicket_pressure"], x["time_urgency"], x["momentum_pressure"]
    d, target, stakes = x["death_over_pressure"], x["target_difficulty"], x["match_stakes"]
    # Nonlinear interaction terms intentionally make the synthetic problem more than a linear copy of PI.
    score = 100 * (
        0.20 * r + 0.16 * w + 0.12 * t + 0.16 * m + 0.09 * d +
        0.10 * target + 0.10 * stakes + 0.07 * (r * d) + 0.05 * (w * d) +
        0.05 * (m * stakes)
    )
    rng = np.random.default_rng(seed)
    score = score + rng.normal(0, 2.0, size=len(score))
    return pd.Series(np.clip(score, 0, 100), index=df.index, name="synthetic_teacher_score")


def generate_synthetic_scenarios(n: int = 40000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scheduled = np.where(rng.random(n) < 0.06, rng.choice([60, 90, 120], size=n), 120)
    balls = np.array([rng.integers(1, int(s) + 1) for s in scheduled])
    req_rate = np.clip(rng.gamma(shape=3.0, scale=3.5, size=n), 0.5, 30)
    wickets = rng.integers(0, 11, size=n)
    recent = np.clip(rng.normal(loc=7.8, scale=3.8, size=n), 0, 24)
    target_difficulty = rng.beta(2.0, 2.0, size=n)
    stakes = rng.choice([0.15, 0.35, 0.55, 0.90, 1.0], size=n, p=[0.35, 0.25, 0.15, 0.15, 0.10])
    is_t20i = rng.integers(0, 2, size=n)
    df = pd.DataFrame({
        "required_run_rate": req_rate,
        "wickets_remaining": wickets,
        "balls_remaining": balls,
        "scheduled_balls": scheduled,
        "recent_run_rate_12": recent,
        "target_difficulty": target_difficulty,
        "match_stakes": stakes,
        "death_over_pressure": np.clip(1 - balls / 30.0, 0, 1),
        "is_t20i": is_t20i,
    })
    df["synthetic_teacher_score"] = synthetic_teacher_score(df, seed=seed + 1)
    df["synthetic_pressure_class"] = pd.cut(
        df["synthetic_teacher_score"], bins=[-0.001, 30, 60, 80, 100],
        labels=CLASS_NAMES, include_lowest=True,
    ).astype(str)
    df["synthetic_pressure_class_id"] = pd.Categorical(
        df["synthetic_pressure_class"], categories=CLASS_NAMES, ordered=True
    ).codes
    return df


def train_synthetic_xgb(
    synthetic: pd.DataFrame,
    random_state: int = 42,
) -> tuple[XGBClassifier, dict, pd.DataFrame, dict]:
    usable = synthetic.dropna(subset=FEATURE_COLUMNS + ["synthetic_pressure_class_id"]).copy()
    X = usable[FEATURE_COLUMNS]
    y = usable["synthetic_pressure_class_id"].astype(int)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, usable.index, test_size=0.22, random_state=random_state, stratify=y
    )
    model = XGBClassifier(
        n_estimators=450, max_depth=5, learning_rate=0.045,
        subsample=0.85, colsample_bytree=0.90, min_child_weight=4,
        reg_lambda=2.0, objective="multi:softprob", num_class=4,
        eval_metric="mlogloss", random_state=random_state, n_jobs=-1,
        tree_method="hist", verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    proba = model.predict_proba(X_test)
    pred = np.argmax(proba, axis=1)
    test_rows = usable.loc[idx_test].copy()
    test_rows["xgb_predicted_class_id"] = pred
    test_rows["xgb_predicted_class"] = [CLASS_NAMES[i] for i in pred]
    test_rows["xgb_predicted_score"] = proba @ CLASS_MIDPOINTS
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "teacher_score_mae_vs_class_midpoint": float(mean_absolute_error(test_rows["synthetic_teacher_score"], test_rows["xgb_predicted_score"])),
        "teacher_score_r2_vs_class_midpoint": float(r2_score(test_rows["synthetic_teacher_score"], test_rows["xgb_predicted_score"])),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "classification_report": classification_report(y_test, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0,1,2,3]).tolist(),
    }
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    split = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    return model, metrics, test_rows, {"importance": importance, "split": split}


def score_with_xgb(model: XGBClassifier, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    X = out[FEATURE_COLUMNS].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        raise ValueError("XGBoost input contains missing feature values")
    proba = model.predict_proba(X)
    pred = proba.argmax(axis=1)
    out["xgb_pressure_score"] = proba @ CLASS_MIDPOINTS
    out["xgb_pressure_class"] = [CLASS_NAMES[i] for i in pred]
    for i, name in enumerate(CLASS_NAMES):
        out[f"xgb_prob_{name.lower()}"] = proba[:, i]
    return out