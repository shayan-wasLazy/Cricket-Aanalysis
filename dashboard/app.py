from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import spearmanr
import streamlit as st
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cricket_pressure.pressure import DEFAULT_WEIGHTS
from cricket_pressure.synthetic import FEATURE_COLUMNS, CLASS_NAMES, CLASS_MIDPOINTS
from cricket_pressure.player_rankings import (
    DEFAULT_RRF_WEIGHTS, DEFAULT_RRF_K, DEFAULT_RRF_MIN_MODEL_SUPPORT,
    build_weighted_rrf_ensemble,
)

st.set_page_config(page_title="SupeRR Pressure Lab", page_icon="🏏", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1.35rem;padding-bottom:2rem;max-width:1550px;}
[data-testid="stMetric"] {background:#111827;border:1px solid #263244;padding:12px 15px;border-radius:12px;}
[data-testid="stMetricLabel"] {color:#9ca3af;}
[data-testid="stMetricValue"] {color:#f9fafb;}
.hero {padding:20px 22px;border:1px solid #263244;border-radius:16px;background:linear-gradient(110deg,#111827,#18283a);margin-bottom:18px;}
.hero h1 {margin:0 0 6px 0;font-size:2rem;}
.hero p,.muted {color:#9ca3af;}
</style>
<div class="hero"><h1>🏏 SupeRR Pressure Lab</h1><p>Pressure Index, synthetic XGBoost, player ranking stability, score audits and historical win-probability validation.</p></div>
""", unsafe_allow_html=True)

OUT = ROOT / "outputs"
PROC = ROOT / "data" / "processed" / "real_pressure_comparison.csv.gz"
MODEL_PATH = ROOT / "models" / "synthetic_pressure_xgb.json"
METRICS_PATH = OUT / "metrics" / "experiment_metrics.json"
OUTCOME_METRICS_PATH = OUT / "metrics" / "historical_win_probability_metrics.json"

if not PROC.exists() or not MODEL_PATH.exists():
    st.warning("Analysis outputs are not available yet. Run python run_analysis.py from the project root.")
    st.code("pip install -r requirements.txt\npython run_analysis.py\nstreamlit run dashboard/app.py", language="bash")
    st.stop()

def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_data():
    return {
        "real": pd.read_csv(PROC, low_memory=False),
        "players": _read_optional(OUT / "player_pressure_comparison.csv"),
        "rankings": _read_optional(OUT / "player_rankings.csv"),
        "top10": _read_optional(OUT / "player_top10_comparison.csv"),
        "by_season": _read_optional(OUT / "match_format_season_summary.csv"),
        "sensitivity": _read_optional(OUT / "weight_sensitivity.csv"),
        "scenario": _read_optional(OUT / "scenario_lab.csv"),
        "importance": _read_optional(OUT / "xgb_feature_importance.csv"),
        "disagreements": _read_optional(OUT / "match_disagreement_summary.csv"),
        "delivery_disagreements": _read_optional(OUT / "delivery_disagreements.csv"),
        "score_audit": _read_optional(OUT / "score_reconciliation_audit.csv"),
        "innings_audit": _read_optional(OUT / "innings_validation.csv"),
        "outcome_calibration": _read_optional(OUT / "historical_win_probability_calibration.csv"),
        "outcome_recency": _read_optional(OUT / "historical_win_probability_recency_comparison.csv"),
        "outcome_importance": _read_optional(OUT / "historical_win_probability_feature_importance.csv"),
        "outcome_predictions": _read_optional(ROOT / "data" / "processed" / "historical_win_probability_predictions.csv.gz"),
        "rr_matches": _read_optional(OUT / "rr_intelligence" / "rr_match_summary.csv"),
        "rr_pressure_points": _read_optional(OUT / "rr_intelligence" / "rr_pressure_points.csv"),
        "rr_bowling_pressure_points": _read_optional(OUT / "rr_intelligence" / "rr_bowling_pressure_points.csv"),
        "rr_pressure_windows": _read_optional(OUT / "rr_intelligence" / "rr_pressure_windows.csv"),
        "rr_phase_gaps": _read_optional(OUT / "rr_intelligence" / "rr_phase_gaps.csv"),
        "rr_player_fit": _read_optional(OUT / "rr_intelligence" / "rr_player_fit_shortlist.csv"),
        "rr_phase_rankings": _read_optional(OUT / "rr_intelligence" / "rr_phase_player_rankings.csv"),
        "rr_frequent_xi": _read_optional(OUT / "rr_intelligence" / "rr_frequent_xi.csv"),
        "rr_frequent_xi_options": _read_optional(OUT / "rr_intelligence" / "rr_frequent_xi_phase_options.csv"),
        "rr_xi_review": _read_optional(OUT / "rr_intelligence" / "rr_xi_pressure_weakness_review.csv"),
        "rr_replacements": _read_optional(OUT / "rr_intelligence" / "rr_xi_replacement_shortlist.csv"),

    }

@st.cache_resource(show_spinner=False)
def load_synthetic_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model

@st.cache_data(show_spinner=False)
def load_json(path: str):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

data = load_data()
df = data["real"]
model = load_synthetic_model()
metrics = load_json(str(METRICS_PATH))
outcome_metrics = load_json(str(OUTCOME_METRICS_PATH))

with st.sidebar:
    st.header("Explore historical states")
    formats = sorted(df["match_format"].dropna().astype(str).unique().tolist())
    selected_formats = st.multiselect("Format", formats, default=formats)
    seasons = sorted(df["season"].dropna().astype(str).unique().tolist())
    selected_seasons = st.multiselect("Season", seasons, default=seasons)
    st.divider()
    st.caption("Baseline PI weights")
    for key, value in DEFAULT_WEIGHTS.items():
        st.write(f"{key.replace('_', ' ').title()}: **{value:.1%}**")
    st.divider()
    st.caption("Synthetic XGBoost reproduces a synthetic teacher rule. Historical outcome XGBoost is trained separately on match winners.")

view = df.loc[
    df["match_format"].astype(str).isin(selected_formats)
    & df["season"].astype(str).isin(selected_seasons)
].copy()
if view.empty:
    st.info("No rows match the selected filters.")
    st.stop()

st.markdown("### Historical pressure overview")
a, b, c, d, e = st.columns(5)
a.metric("Chase deliveries", f"{len(view):,}")
b.metric("Matches", f"{view['match_id'].nunique():,}")
c.metric("Mean PI", f"{view['pressure_index'].mean():.1f}")
d.metric("Mean synthetic XGB", f"{view['xgb_pressure_score'].mean():.1f}")
e.metric("PI ↔ synthetic XGB Spearman", f"{view[['pressure_index','xgb_pressure_score']].corr(method='spearman').iloc[0,1]:.3f}")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("PI and synthetic-trained XGBoost")
    sample = view.sample(min(5000, len(view)), random_state=11)
    fig = px.scatter(
        sample, x="pressure_index", y="xgb_pressure_score", color="match_format",
        hover_data=["match_id", "season", "required_run_rate", "wickets_remaining", "balls_remaining"],
        opacity=.55, labels={"pressure_index":"PI heuristic", "xgb_pressure_score":"Synthetic-trained XGBoost"},
    )
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Pressure class mix")
    counts = pd.concat([
        view["pressure_category"].astype(str).value_counts().rename("PI"),
        view["xgb_pressure_class"].astype(str).value_counts().rename("XGBoost"),
    ], axis=1).reindex(CLASS_NAMES).fillna(0).reset_index(names="Pressure class")
    fig = px.bar(counts, x="Pressure class", y=["PI","XGBoost"], barmode="group", height=420)
    fig.update_layout(margin=dict(l=10,r=10,t=20,b=10), legend_title="Model")
    st.plotly_chart(fig, use_container_width=True)

tabs = st.tabs([
    "Synthetic diagnostics", "Scenario playground", "Weight lab",
    "Match disagreement", "Player rankings", "Historical win model", "Score audit", "Recency analysis",
    "RRF ensemble shortlist", "RR squad intelligence", "PI/XGB phase selector & RR XI",
    "Final submission + RR XI decisions",
])

with tabs[0]:
    st.subheader("Synthetic holdout — XGBoost vs PI")
    st.caption("These metrics evaluate both methods against the generated synthetic labels. They do not validate psychological pressure in real matches.")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("PI accuracy", f"{metrics.get('pi_accuracy', float('nan')):.3f}")
    m2.metric("XGBoost accuracy", f"{metrics.get('xgb_accuracy', float('nan')):.3f}")
    m3.metric("PI macro-F1", f"{metrics.get('pi_macro_f1', float('nan')):.3f}")
    m4.metric("XGBoost macro-F1", f"{metrics.get('xgb_macro_f1', float('nan')):.3f}")
    m5,m6,m7 = st.columns(3)
    m5.metric("Test scenarios", f"{metrics.get('n_test', 0):,}")
    m6.metric("PI teacher-score MAE", f"{metrics.get('pi_teacher_score_mae', float('nan')):.2f}")
    m7.metric("XGB teacher-score MAE", f"{metrics.get('xgb_teacher_score_mae_vs_class_midpoint', float('nan')):.2f}")
    l,r = st.columns(2)
    with l:
        if not data["importance"].empty:
            fig = px.bar(data["importance"].sort_values("importance"), x="importance", y="feature",
                         orientation="h", title="Feature importance in synthetic task")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    with r:
        cm = np.asarray(metrics.get("confusion_matrix", []), dtype=float)
        cm_path = OUT / "metrics" / "synthetic_xgb_confusion_matrix.csv"
        if cm.size == 0 and cm_path.exists():
            cm = pd.read_csv(cm_path, header=None).to_numpy()
        if cm.shape == (4,4):
            fig = px.imshow(cm, x=CLASS_NAMES, y=CLASS_NAMES, text_auto=True,
                            labels={"x":"Predicted class","y":"Synthetic teacher class","color":"Scenarios"},
                            color_continuous_scale="Blues", title="Synthetic holdout confusion matrix")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Confusion matrix missing. Rerun python run_analysis.py after pulling the latest code.")
    st.warning("XGBoost can score better because the synthetic teacher deliberately contains nonlinear interactions. This is rule-learning performance, not evidence of better real-world pressure measurement.")

with tabs[1]:
    st.subheader("Scenario playground")
    st.caption("Enter a hypothetical chase state. The PI and XGBoost outputs are both heuristic/synthetic-model outputs, not match win probabilities.")
    x1,x2,x3 = st.columns(3)
    with x1:
        rrr = st.slider("Required run rate", 0.5, 30.0, 13.0, 0.5)
        wickets_left = st.slider("Wickets remaining", 0, 10, 5)
    with x2:
        overs_total = st.selectbox("Scheduled overs", [10, 15, 20], index=2)
        overs_left = st.slider("Overs remaining", 0.5, float(overs_total), min(10.0,float(overs_total)), 0.5)
    with x3:
        recent_rr = st.slider("Recent run rate", 0.0, 24.0, 9.0, 0.5)
        target_pct = st.slider("Target difficulty percentile", 0.0, 1.0, 0.70, 0.05)
        stakes = st.slider("Match stakes", 0.0, 1.0, 0.25, 0.05)
    balls_left = max(1, int(round(overs_left * 6)))
    scheduled_balls = int(overs_total * 6)
    row = pd.DataFrame([{
        "required_run_rate":rrr, "wickets_remaining":wickets_left,
        "balls_remaining":balls_left, "scheduled_balls":scheduled_balls,
        "recent_run_rate_12":recent_rr, "target_difficulty":target_pct,
        "match_stakes":stakes, "death_over_pressure":float(np.clip(1-balls_left/30,0,1)),
        "is_t20i":0,
    }])
    row["rrr_pressure"] = np.clip(rrr/12,0,1)
    row["wicket_pressure"] = np.clip(1-wickets_left/10,0,1)
    row["time_urgency"] = np.clip(1-balls_left/scheduled_balls,0,1)
    row["momentum_pressure"] = np.clip(max(rrr-recent_rr,0)/12,0,1)
    pi_score = 100 * sum(row[k].iloc[0]*w for k,w in DEFAULT_WEIGHTS.items())
    probs = model.predict_proba(row[FEATURE_COLUMNS])[0]
    xgb_score = float(probs @ CLASS_MIDPOINTS)
    q1,q2,q3 = st.columns(3)
    q1.metric("PI heuristic", f"{pi_score:.1f}/100")
    q2.metric("Synthetic XGB score", f"{xgb_score:.1f}/100")
    q3.metric("XGB class", CLASS_NAMES[int(probs.argmax())])
    fig = px.bar(pd.DataFrame({"Class":CLASS_NAMES,"Probability":probs}), x="Class", y="Probability",
                 range_y=[0,1], title="Synthetic XGBoost class probabilities")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Death-over pressure component: {row['death_over_pressure'].iloc[0]:.2f} · {balls_left} legal balls remaining")

with tabs[2]:
    st.subheader("PI weight sensitivity lab")
    st.caption("Sliders are normalized to 100%. Compare the score distribution and rank ordering with the current PI.")
    keys = list(DEFAULT_WEIGHTS)
    cols = st.columns(4)
    chosen = {}
    for i,key in enumerate(keys):
        with cols[i % 4]:
            chosen[key] = st.slider(key.replace("_"," ").title(), 0.0, 0.50,
                                    float(DEFAULT_WEIGHTS[key]), 0.01, key=f"weight_{key}")
    total = sum(chosen.values())
    if total <= 0:
        st.error("At least one weight must be above zero.")
    else:
        normalized = {k:v/total for k,v in chosen.items()}
        adjusted = 100 * sum(view[k].fillna(0)*w for k,w in normalized.items())
        rho = float(spearmanr(view["pressure_index"], adjusted).statistic)
        q1,q2,q3 = st.columns(3)
        q1.metric("Adjusted mean PI", f"{adjusted.mean():.2f}")
        q2.metric("Spearman vs baseline", f"{rho:.4f}")
        q3.metric("Mean absolute change", f"{(adjusted-view['pressure_index']).abs().mean():.2f}")
        st.write("Normalized weights", {k:f"{v:.1%}" for k,v in normalized.items()})
        fig = px.histogram(pd.DataFrame({"Baseline PI":view["pressure_index"],"Adjusted PI":adjusted}),
                           barmode="overlay",opacity=.6,title="Baseline versus adjusted PI distribution")
        st.plotly_chart(fig, use_container_width=True)
    if not data["sensitivity"].empty:
        st.subheader("Precomputed weight scenarios")
        st.dataframe(data["sensitivity"], use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Match-by-match disagreement")
    st.caption("Matches are ranked by average absolute difference between PI and the synthetic-trained XGBoost score. This identifies cases worth manual review; it does not tell us which model is correct.")
    disagreements = data["disagreements"]
    if disagreements.empty:
        st.info("No match disagreement file found. Run the latest analysis pipeline.")
    else:
        st.dataframe(disagreements.head(30), use_container_width=True, hide_index=True)
        visible_match_ids = set(view["match_id"].dropna().astype(int).tolist())
        filtered_disagreements = disagreements.loc[
            pd.to_numeric(disagreements["match_id"], errors="coerce").isin(visible_match_ids)
        ]
        opts = filtered_disagreements["match_id"].dropna().astype(int).tolist()
        if not opts:
            st.info("No disagreement matches remain after the current format/season filters.")
            st.stop()
        selected_match = st.selectbox("Inspect match ID", opts, key="disagreement_match")
        match_rows = view.loc[view["match_id"].eq(selected_match)].sort_values("innings_ball_num")
        if not match_rows.empty:
            m1,m2,m3 = st.columns(3)
            m1.metric("Mean absolute gap", f"{match_rows['absolute_pi_xgb_gap'].mean():.2f}" if "absolute_pi_xgb_gap" in match_rows else f"{(match_rows['pressure_index']-match_rows['xgb_pressure_score']).abs().mean():.2f}")
            m2.metric("PI average", f"{match_rows['pressure_index'].mean():.1f}")
            m3.metric("XGB average", f"{match_rows['xgb_pressure_score'].mean():.1f}")
            fig = px.line(match_rows, x="innings_ball_num", y=["pressure_index","xgb_pressure_score"],
                          labels={"innings_ball_num":"Delivery sequence","value":"Pressure score","variable":"Model"},
                          title=f"Pressure trajectory — match {selected_match}")
            st.plotly_chart(fig, use_container_width=True)
            show_cols = [c for c in ["innings_ball_num","score_before","runs_required","required_run_rate",
                                     "wickets_remaining","balls_remaining","pressure_index","xgb_pressure_score"]
                         if c in match_rows.columns]
            detail = match_rows.copy()
            detail["absolute_gap"] = (detail["pressure_index"]-detail["xgb_pressure_score"]).abs()
            st.caption("Most divergent deliveries in the selected match")
            st.dataframe(detail.sort_values("absolute_gap",ascending=False)[show_cols+["absolute_gap"]].head(15),
                         use_container_width=True, hide_index=True)
        if not data["delivery_disagreements"].empty:
            with st.expander("Largest delivery-level disagreements across all matches"):
                st.dataframe(data["delivery_disagreements"].head(100), use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("High-pressure player rankings")
    st.caption("Performance within high-pressure deliveries (score ≥60). Batter metric: runs/100 balls. Bowler metric: credited wickets/100 balls. Rates are shrunk toward the role-wide rate; they are not yet clutch uplift versus each player's normal baseline.")
    rankings = data["rankings"]
    if rankings.empty:
        st.info("Player rankings have not been generated. Run python run_analysis.py.")
    else:
        c1,c2,c3 = st.columns(3)
        with c1:
            role = st.selectbox("Player role", ["Batter","Bowler"])
        with c2:
            max_d = int(max(100, rankings["pressure_deliveries"].max()))
            min_deliveries = st.number_input("Minimum pressure deliveries", min_value=0, max_value=max_d,
                                             value=100, step=25)
        with c3:
            max_m = int(max(10, rankings["pressure_matches"].max()))
            min_matches = st.number_input("Minimum pressure matches", min_value=0, max_value=max_m,
                                          value=10, step=1)
        methods = sorted(rankings["scoring_method"].dropna().unique().tolist(),
                         key=lambda x: list(["PI baseline","PI RRR-heavy","PI momentum-heavy","PI stakes-heavy","XGBoost"]).index(x)
                         if x in ["PI baseline","PI RRR-heavy","PI momentum-heavy","PI stakes-heavy","XGBoost"] else 99)
        selected_methods = st.multiselect("Models to compare", methods, default=methods)
        half_lives = ["unweighted", "180d", "365d", "540d", "730d"]
        selected_half_life = st.selectbox(
            "Recency setting",
            half_lives,
            index=1,
            format_func=lambda x: {
                "unweighted":"No decay",
                "180d":"180-day half-life (strong recency)",
                "365d":"365-day half-life",
                "540d":"540-day half-life",
                "730d":"730-day half-life",
            }[x],
        )
        activity_window = st.selectbox(
            "Recent pressure activity",
            ["365d", "730d", "All history"],
            index=0,
            format_func=lambda x: {
                "365d":"Pressure appearance in last 365 days",
                "730d":"Pressure appearance in last 730 days",
                "All history":"No activity filter",
            }[x],
            help="Filters out players whose latest high-pressure appearance is too old for current selection.",
        )
        p = rankings.loc[
            rankings["role"].eq(role)
            & rankings["scoring_method"].isin(selected_methods)
            & rankings["half_life"].eq(selected_half_life)
        ].copy()
        p["eligible_current"] = (
            p["pressure_deliveries"].ge(min_deliveries)
            & p["pressure_matches"].ge(min_matches)
        )
        if activity_window != "All history" and "days_since_last_pressure_match" in p:
            max_age_days = 365 if activity_window == "365d" else 730
            p["eligible_current"] &= p["days_since_last_pressure_match"].le(max_age_days)
        reference_dates = rankings["recency_reference_date"].dropna().astype(str).unique()
        if len(reference_dates):
            st.caption(
                f"Player recency is aged from **{reference_dates[0][:10]} UTC**, not from the newest match in the data pack."
            )
        p["current_rank"] = np.nan
        for method, idx in p.loc[p["eligible_current"]].groupby("scoring_method").groups.items():
            p.loc[idx, "current_rank"] = p.loc[idx, "shrunk_per_100"].rank(method="min", ascending=False)
        top = p.loc[p["eligible_current"] & p["current_rank"].le(10)].sort_values(["scoring_method","current_rank"])
        k1,k2,k3 = st.columns(3)
        k1.metric("Eligible players", f"{p['eligible_current'].sum():,}")
        base_names = set(top.loc[top["scoring_method"].eq("PI baseline"),"player_id"])
        if base_names:
            overlap = len(base_names.intersection(set(top.loc[top["scoring_method"].eq("XGBoost"),"player_id"])))
            k2.metric("PI / XGB top-10 overlap", f"{overlap}/10")
        else:
            k2.metric("PI / XGB top-10 overlap", "—")
        k3.metric("Methods compared", f"{len(selected_methods)}")
        show_columns = [
            "scoring_method", "role", "current_rank", "player_name",
            "pressure_deliveries", "pressure_matches",
            "last_pressure_match_date", "days_since_last_pressure_match",
            "recent_pressure_deliveries_365d", "observed_per_100",
            "shrunk_per_100", "eligible_current",
        ]
        show_columns = [column for column in show_columns if column in top.columns]
        st.dataframe(
            top[show_columns].rename(columns={
                "current_rank": "Rank",
                "player_name": "Player",
                "pressure_deliveries": "All-time pressure deliveries",
                "pressure_matches": "All-time pressure matches",
                "last_pressure_match_date": "Last pressure appearance",
                "days_since_last_pressure_match": "Days since pressure appearance",
                "recent_pressure_deliveries_365d": "Pressure deliveries in last 365d",
                "observed_per_100": "Raw rate / 100",
                "shrunk_per_100": "Recency-adjusted rate / 100",
            }),
            use_container_width=True, hide_index=True,
        )
        if p["eligible_current"].sum() == 0:
            st.warning(
                "No players pass the current activity window and sample gates. "
                "Try the 730-day or All history activity filter, or check whether the dataset is current."
            )
        if not top.empty:
            fig = px.bar(top, x="player_name", y="shrunk_per_100", color="scoring_method",
                         barmode="group", facet_col="scoring_method",
                         title=f"Top eligible {role.lower()}s by high-pressure performance",
                         labels={"shrunk_per_100":"Shrunk rate per 100 pressure deliveries","player_name":"Player"})
            fig.update_layout(height=520, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("All player samples, including ineligible players"):
            st.dataframe(p.sort_values(["scoring_method","eligible_current","shrunk_per_100"],
                                       ascending=[True,False,False]),
                         use_container_width=True, hide_index=True)


with tabs[7]:
    st.subheader("Recency weighting laboratory")
    st.caption("Player rankings now age observations from the current UTC date, so inactive players do not stay artificially recent when the source data pack is stale. The historical win-probability experiment continues to use its separate chronological training reference.")
    recency_table = data["outcome_recency"]
    if recency_table.empty:
        st.info("Recency comparison is not available. Run the latest python run_analysis.py.")
    else:
        st.markdown("#### Historical win-probability model")
        st.caption("All models share the same chronological match-level train/test split. Recency weights are used only for training observations. The held-out matches are evaluated with identical match-balanced weights for every model.")
        pretty = recency_table.copy()
        pretty["Brier score"] = pretty["brier_score_match_weighted"]
        pretty["ROC-AUC"] = pretty["roc_auc_match_weighted"]
        pretty["Log loss"] = pretty["log_loss_match_weighted"]
        st.dataframe(pretty[["half_life","train_matches","test_matches","Brier score","ROC-AUC","Log loss","cutoff_date_utc"]],
                     use_container_width=True, hide_index=True)
        l,r = st.columns(2)
        with l:
            fig = px.line(recency_table, x="half_life", y="brier_score_match_weighted", markers=True,
                          title="Out-of-time Brier score by training recency",
                          labels={"half_life":"Training recency","brier_score_match_weighted":"Brier score (lower is better)"})
            st.plotly_chart(fig, use_container_width=True)
        with r:
            fig = px.line(recency_table, x="half_life", y="roc_auc_match_weighted", markers=True,
                          title="Out-of-time ROC-AUC by training recency",
                          labels={"half_life":"Training recency","roc_auc_match_weighted":"ROC-AUC (higher is better)"})
            st.plotly_chart(fig, use_container_width=True)

    rankings = data["rankings"]
    if not rankings.empty:
        st.markdown("#### Player ranking stability")
        st.caption("All player-ranking half-lives are anchored to the current UTC date. A 180-day half-life is the new recency-forward default; longer half-lives remain available for sensitivity checks.")
        role_for_stability = st.selectbox("Role for recency comparison", ["Batter","Bowler"], key="recency_role")
        methods_for_stability = sorted(rankings["scoring_method"].dropna().unique().tolist())
        method_for_stability = st.selectbox("Pressure definition", methods_for_stability, key="recency_method")
        player_min_deliveries = st.number_input("Minimum pressure deliveries for stability",
                                                min_value=0, max_value=100000, value=100, step=25,
                                                key="recency_min_deliveries")
        player_min_matches = st.number_input("Minimum pressure matches for stability",
                                             min_value=0, max_value=10000, value=10, step=1,
                                             key="recency_min_matches")
        rp = rankings.loc[
            rankings["role"].eq(role_for_stability)
            & rankings["scoring_method"].eq(method_for_stability)
            & rankings["pressure_deliveries"].ge(player_min_deliveries)
            & rankings["pressure_matches"].ge(player_min_matches)
        ].copy()
        if rp.empty:
            st.info("No eligible players under these sample thresholds.")
        else:
            rp["rank_under_current_thresholds"] = np.nan
            for half, idx in rp.groupby("half_life").groups.items():
                rp.loc[idx, "rank_under_current_thresholds"] = (
                    rp.loc[idx, "shrunk_per_100"].rank(method="min", ascending=False)
                )
            top_sets = {}
            for half in ["unweighted","180d","365d","540d","730d"]:
                top_sets[half] = set(
                    rp.loc[rp["half_life"].eq(half)]
                    .nsmallest(10, "rank_under_current_thresholds")["player_id"].dropna()
                )
            overlap_df = pd.DataFrame([
                {
                    "half_life": half,
                    "top10_overlap_vs_unweighted": len(top_sets["unweighted"] & top_sets[half]),
                    "eligible_players": int(rp.loc[rp["half_life"].eq(half), "player_id"].nunique()),
                }
                for half in ["unweighted","180d","365d","540d","730d"]
            ])
            st.dataframe(overlap_df, use_container_width=True, hide_index=True)
            movement = rp.pivot_table(index=["player_id","player_name"], columns="half_life",
                                      values="rank_under_current_thresholds", aggfunc="min").reset_index()
            if "unweighted" in movement.columns:
                st.caption("Rank movement by half-life. Lower rank is better; a missing rank means the player is outside the eligible top-ranked group.")
                st.dataframe(movement.sort_values("unweighted").head(50), use_container_width=True, hide_index=True)

    st.markdown("#### Recency-weight curve")
    age = np.arange(0, 4 * 365 + 1, 30)
    curve = pd.DataFrame({"age_days": age})
    for half in [180,365,540,730]:
        curve[f"{half}-day half-life"] = np.power(2.0, -age / half)
    curve_long = curve.melt(id_vars="age_days", var_name="Half-life", value_name="Relative weight")
    fig = px.line(curve_long, x="age_days", y="Relative weight", color="Half-life",
                  title="How historical matches are discounted",
                  labels={"age_days":"Age of match (days)","Relative weight":"Contribution multiplier"})
    st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.subheader("Historical win-probability model")
    st.caption("This model is separate from the synthetic pressure classifier. It predicts whether the second-innings batting team eventually won, using pre-delivery state and a chronological match holdout. The displayed model uses a provisional 540-day training half-life; compare it with no decay, 365 days and 730 days in the Recency analysis tab.")
    if not outcome_metrics:
        st.info("Historical outcome model outputs not found. Run the latest python run_analysis.py.")
    else:
        q1,q2,q3,q4 = st.columns(4)
        q1.metric("Match-weighted ROC-AUC", f"{outcome_metrics.get('roc_auc_match_weighted', float('nan')):.3f}")
        q2.metric("Match-weighted Brier", f"{outcome_metrics.get('brier_score_match_weighted', float('nan')):.3f}")
        q3.metric("Baseline Brier", f"{outcome_metrics.get('baseline_brier_score_match_weighted', float('nan')):.3f}")
        q4.metric("Match-weighted log loss", f"{outcome_metrics.get('log_loss_match_weighted', float('nan')):.3f}")
        q5,q6,q7 = st.columns(3)
        q5.metric("Train matches", f"{outcome_metrics.get('train_matches',0):,}")
        q6.metric("Test matches", f"{outcome_metrics.get('test_matches',0):,}")
        q7.metric("Test deliveries", f"{outcome_metrics.get('test_deliveries',0):,}")
        st.write(f"Chronological cutoff: **{outcome_metrics.get('cutoff_date_utc','unavailable')}**")
        l,r = st.columns(2)
        with l:
            cal = data["outcome_calibration"]
            if not cal.empty:
                fig = px.line(cal, x="mean_predicted_probability", y="observed_win_rate",
                              markers=True, range_x=[0,1], range_y=[0,1],
                              title="Win probability calibration",
                              labels={"mean_predicted_probability":"Mean predicted win probability",
                                      "observed_win_rate":"Observed win rate"})
                fig.add_shape(type="line", x0=0,y0=0,x1=1,y1=1,line=dict(dash="dash",color="gray"))
                st.plotly_chart(fig, use_container_width=True)
        with r:
            imp = data["outcome_importance"]
            if not imp.empty:
                fig = px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h",
                             title="Historical outcome model feature importance")
                st.plotly_chart(fig, use_container_width=True)
        st.warning("This is an empirical outcome benchmark, not psychological pressure ground truth. A badly calibrated win-probability curve should not be converted directly into a pressure score.")
        if not data["outcome_predictions"].empty:
            with st.expander("Held-out delivery predictions"):
                st.dataframe(data["outcome_predictions"].head(100), use_container_width=True, hide_index=True)

with tabs[6]:
    st.subheader("Score reconstruction and innings audit")
    st.caption("The source pack does not provide a separate official scorecard table. This audit checks innings integrity and only compares run margins where result text supports it and no bye/legbye runs remain unresolved.")
    audit = data["score_audit"]
    innings_audit = data["innings_audit"]
    if audit.empty:
        st.info("Score audit not generated. Run python run_analysis.py.")
    else:
        eligible = int(audit["run_margin_audit_eligible"].sum())
        consistent = int(audit["run_margin_audit_status"].eq("consistent with reported margin").sum())
        unresolved = int(audit["unresolved_extra_deliveries"].gt(0).sum())
        q1,q2,q3,q4 = st.columns(4)
        q1.metric("Matches audited", f"{len(audit):,}")
        q2.metric("Comparable run margins", f"{eligible:,}")
        q3.metric("Margins consistent", f"{consistent:,}")
        q4.metric("Matches with unresolved extras", f"{unresolved:,}")
        st.warning("Matches with unresolved bye/legbye scoring are explicitly marked provisional; their score-margin differences are not used as a pass/fail check.")
        status_options = ["All"] + sorted(audit["run_margin_audit_status"].dropna().unique().tolist())
        status_filter = st.selectbox("Audit status", status_options)
        filtered = audit if status_filter == "All" else audit.loc[audit["run_margin_audit_status"].eq(status_filter)]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        if not innings_audit.empty:
            with st.expander("Innings-level legal-ball and wicket checks"):
                st.dataframe(innings_audit, use_container_width=True, hide_index=True)


with tabs[8]:
    st.subheader("Weighted Reciprocal Rank Fusion — candidate shortlist")
    st.caption(
        "RRF combines independent player rank lists with unequal model weights. "
        "This produces a research shortlist for human review; it does not automatically choose the final Top 10."
    )
    rankings = data["rankings"]
    if rankings.empty:
        st.info("Player rankings have not been generated. Run python refresh_player_rankings.py.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            rrf_role = st.selectbox("Ensemble role", ["Batter", "Bowler"], key="rrf_role")
        with c2:
            rrf_min_deliveries = st.number_input(
                "Minimum pressure deliveries", min_value=0, max_value=100000,
                value=100, step=25, key="rrf_min_deliveries",
            )
        with c3:
            rrf_min_matches = st.number_input(
                "Minimum pressure matches", min_value=0, max_value=10000,
                value=10, step=1, key="rrf_min_matches",
            )

        c4, c5, c6 = st.columns(3)
        with c4:
            rrf_half_life = st.selectbox(
                "Recency setting",
                ["180d", "365d", "540d", "730d", "unweighted"],
                index=0, key="rrf_half_life",
            )
        with c5:
            rrf_activity = st.selectbox(
                "Recent pressure activity",
                ["365d", "730d", "All history"],
                index=0, key="rrf_activity",
            )
        with c6:
            rrf_min_support = st.selectbox(
                "Minimum model support",
                [3, 4, 5],
                index=0,
                format_func=lambda x: f"{x} of 5 models",
                key="rrf_min_support",
            )

        rrf_top_n = st.slider(
            "Shortlist size", min_value=10, max_value=30, value=20, step=5,
            key="rrf_top_n",
        )
        max_age = None if rrf_activity == "All history" else (365 if rrf_activity == "365d" else 730)
        ensemble = build_weighted_rrf_ensemble(
            rankings,
            min_deliveries=int(rrf_min_deliveries),
            min_matches=int(rrf_min_matches),
            half_life=rrf_half_life,
            max_days_since_last_match=max_age,
            min_model_support=int(rrf_min_support),
            rrf_k=DEFAULT_RRF_K,
        )
        role_ensemble = ensemble.loc[ensemble["role"].eq(rrf_role)].copy() if not ensemble.empty else pd.DataFrame()

        st.markdown("#### Unequal model weights")
        weight_table = pd.DataFrame([
            {"Scoring method": method, "RRF weight": f"{weight:.0%}"}
            for method, weight in DEFAULT_RRF_WEIGHTS.items()
        ])
        st.dataframe(weight_table, use_container_width=True, hide_index=True)
        st.caption(
            "The synthetic-trained XGBoost receives only 5% because it learns a synthetic teacher rule, "
            "not empirically labelled psychological pressure. The four PI methods are related sensitivity "
            "variants, so their votes are deliberately not equal."
        )
        st.latex(r"\mathrm{RRF}(p)=\sum_m \frac{w_m}{60+\mathrm{rank}_m(p)}")

        if role_ensemble.empty:
            st.warning(
                "No players pass the selected sample, recency and model-support gates. "
                "Try a longer activity window or inspect the minimum-sample settings."
            )
        else:
            shortlist = role_ensemble.head(int(rrf_top_n)).copy()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Eligible ensemble candidates", f"{len(role_ensemble):,}")
            m2.metric("Shortlist size", f"{len(shortlist):,}")
            m3.metric("Median model support", f"{role_ensemble['model_support'].median():.1f}/5")
            m4.metric("Top candidate RRF score", f"{shortlist.iloc[0]['rrf_score']:.5f}")

            rank_columns = [
                "rank_pi_baseline", "rank_pi_rrr_heavy", "rank_pi_momentum_heavy",
                "rank_pi_stakes_heavy", "rank_xgboost",
            ]
            show_columns = [
                "ensemble_rank", "player_name", "rrf_score", "model_support",
                "model_weight_coverage", *rank_columns,
                "min_pressure_deliveries_supported", "min_pressure_matches_supported",
                "last_pressure_match_date", "days_since_last_pressure_match",
            ]
            show_columns = [col for col in show_columns if col in shortlist.columns]
            st.dataframe(
                shortlist[show_columns].rename(columns={
                    "ensemble_rank": "RRF rank",
                    "player_name": "Player",
                    "rrf_score": "Weighted RRF score",
                    "model_support": "Model support / 5",
                    "model_weight_coverage": "Supporting weight",
                    "rank_pi_baseline": "PI baseline rank",
                    "rank_pi_rrr_heavy": "RRR-heavy rank",
                    "rank_pi_momentum_heavy": "Momentum-heavy rank",
                    "rank_pi_stakes_heavy": "Stakes-heavy rank",
                    "rank_xgboost": "Synthetic XGB rank",
                    "min_pressure_deliveries_supported": "Fewest pressure balls in supporting models",
                    "min_pressure_matches_supported": "Fewest pressure matches in supporting models",
                    "last_pressure_match_date": "Last pressure appearance",
                    "days_since_last_pressure_match": "Days since pressure appearance",
                }),
                use_container_width=True,
                hide_index=True,
            )
            fig = px.bar(
                shortlist.sort_values("rrf_score", ascending=True),
                x="rrf_score", y="player_name", orientation="h",
                color="model_support",
                title=f"Weighted RRF shortlist — {rrf_role.lower()}s",
                labels={
                    "rrf_score": "Weighted RRF score",
                    "player_name": "Player",
                    "model_support": "Model support",
                },
            )
            fig.update_layout(height=max(420, 28 * len(shortlist)), margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "Use this shortlist to inspect evidence and role fit before choosing your final ten. "
                "RRF measures cross-model ranking agreement; it is not itself a measure of player quality or pressure uplift."
            )



with tabs[9]:
    st.subheader("Rajasthan Royals — squad intelligence")
    st.caption(
        "RR match diagnostics are measured against the rest of the IPL player pool. "
        "Candidate scores are role/phase percentiles, not auction availability or a final bid recommendation."
    )
    rr_matches = data["rr_matches"]
    rr_batting_pressure = data["rr_pressure_points"]
    rr_bowling_pressure = data["rr_bowling_pressure_points"]
    rr_windows = data["rr_pressure_windows"]
    rr_gaps = data["rr_phase_gaps"]
    rr_fit = data["rr_player_fit"]

    if rr_matches.empty or rr_gaps.empty:
        st.info("RR intelligence outputs are not available yet. Download and process the Cricsheet IPL archive.")
        st.code(
            "python scripts/build_rr_intelligence.py --download --start-year 2022 --end-year 2026",
            language="bash",
        )
    else:
        deliveries_count = int(rr_matches[[
            "innings_1_legal_balls", "innings_2_legal_balls"
        ]].fillna(0).to_numpy().sum()) if {
            "innings_1_legal_balls", "innings_2_legal_balls"
        }.issubset(rr_matches.columns) else 0
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("RR matches", f"{len(rr_matches):,}")
        q2.metric("RR legal deliveries", f"{deliveries_count:,}")
        q3.metric("High batter-pressure balls", f"{len(rr_batting_pressure):,}")
        q4.metric("High bowler-pressure balls", f"{len(rr_bowling_pressure):,}")

        st.markdown("#### RR phase gaps vs the rest of IPL")
        gap_cols = [
            "role", "phase", "team_need_score", "rr_balls",
            "rr_strike_rate", "league_strike_rate",
            "rr_economy", "league_economy",
        ]
        gap_cols = [c for c in gap_cols if c in rr_gaps.columns]
        st.dataframe(
            rr_gaps[gap_cols].sort_values("team_need_score", ascending=False),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Need score is a transparent heuristic based on phase-specific batting strike rate, "
            "dismissal/boundary rates or bowling economy, wicket and dot-ball rates. Inspect raw rates and samples."
        )

        st.markdown("#### Candidate fit by role and phase")
        f1, f2 = st.columns(2)
        with f1:
            fit_role = st.selectbox("Candidate role", ["Batter", "Bowler"], key="rr_fit_role")
        with f2:
            fit_phase = st.selectbox("Target phase", ["Powerplay", "Middle", "Death"], key="rr_fit_phase")
        candidates = rr_fit.loc[
            rr_fit["role"].eq(fit_role) & rr_fit["phase"].eq(fit_phase)
        ].sort_values("auction_fit_score", ascending=False) if not rr_fit.empty else pd.DataFrame()
        if candidates.empty:
            st.warning("No candidate profiles found for this role and phase.")
        else:
            cols = [
                "player_name", "franchises", "auction_fit_score", "player_fit_score",
                "rr_team_need_score", "deliveries", "matches", "pressure_deliveries",
                "pressure_sample_adequate", "availability_status",
            ]
            cols = [c for c in cols if c in candidates.columns]
            st.dataframe(candidates[cols].head(20), use_container_width=True, hide_index=True)
        st.warning(
            "This is a performance-fit shortlist, not an auction-availability list. "
            "Before recommending a purchase, join it to the official 2027 retention/auction pool, "
            "then account for overseas slots, role balance, budget and likely price."
        )

        st.markdown("#### Highest-pressure RR chase windows")
        if rr_windows.empty:
            st.info("No chase windows crossed the current batting-pressure threshold.")
        else:
            st.dataframe(rr_windows.head(20), use_container_width=True, hide_index=True)
        with st.expander("High-pressure batter deliveries"):
            if rr_batting_pressure.empty:
                st.write("No batting-pressure delivery rows.")
            else:
                st.dataframe(rr_batting_pressure.head(200), use_container_width=True, hide_index=True)
        with st.expander("High-pressure bowling deliveries"):
            if rr_bowling_pressure.empty:
                st.write("No bowling-pressure delivery rows.")
            else:
                st.dataframe(rr_bowling_pressure.head(200), use_container_width=True, hide_index=True)
        st.markdown("#### Recent RR match results")
        match_cols = [
            "match_date", "season", "rr_opponent", "rr_result", "winner",
            "venue", "innings_1_runs", "innings_2_runs",
        ]
        match_cols = [c for c in match_cols if c in rr_matches.columns]
        st.dataframe(
            rr_matches[match_cols].sort_values("match_date", ascending=False).head(15),
            use_container_width=True, hide_index=True,
        )




with tabs[10]:
    st.subheader("Phase-wise player selector — PI + XGBoost")
    st.caption(
        "Ranks players by performance in high-pressure deliveries for the selected role and phase. "
        "PI is the primary signal; the synthetic-trained XGBoost contributes a smaller supporting weight."
    )
    phase_rankings = data["rr_phase_rankings"]
    frequent_xi = data["rr_frequent_xi"]
    xi_options = data["rr_frequent_xi_options"]

    selector_tab, xi_tab = st.tabs(["Phase-wise player options", "RR 2026 frequent XI"])

    with selector_tab:
        if phase_rankings.empty:
            st.info("Phase rankings have not been generated. Re-run the RR intelligence build after pulling the latest code.")
            st.code("python scripts/build_rr_intelligence.py --download --start-year 2022 --end-year 2026", language="bash")
        else:
            c1, c2 = st.columns(2)
            with c1:
                selection_role = st.selectbox("Player role", ["Batter", "Bowler"], key="pi_xgb_role")
            with c2:
                selection_phase = st.selectbox("Match phase", ["Powerplay", "Middle", "Death"], key="pi_xgb_phase")
            c3, c4, c5 = st.columns(3)
            with c3:
                selection_pool = st.selectbox(
                    "Player pool", ["All IPL players", "RR 2026 season XI only"], key="pi_xgb_pool"
                )
            with c4:
                min_pressure_balls = st.number_input(
                    "Minimum pressure deliveries per model", min_value=0, max_value=10000,
                    value=20, step=5, key="pi_xgb_min_balls",
                )
            with c5:
                min_pressure_matches = st.number_input(
                    "Minimum pressure matches per model", min_value=0, max_value=1000,
                    value=3, step=1, key="pi_xgb_min_matches",
                )

            candidates = phase_rankings.loc[
                phase_rankings["role"].eq(selection_role)
                & phase_rankings["phase"].eq(selection_phase)
            ].copy()
            if selection_pool == "RR 2026 season XI only":
                xi_ids = set(frequent_xi["player_id"].astype(str)) if not frequent_xi.empty else set()
                candidates = candidates.loc[candidates["player_key"].astype(str).isin(xi_ids)]

            pi_ok = (
                candidates["pi_performance_score"].notna()
                & candidates["pi_pressure_deliveries"].fillna(0).ge(min_pressure_balls)
                & candidates["pi_pressure_matches"].fillna(0).ge(min_pressure_matches)
            )
            xgb_ok = (
                candidates["xgb_performance_score"].notna()
                & candidates["xgb_pressure_deliveries"].fillna(0).ge(min_pressure_balls)
                & candidates["xgb_pressure_matches"].fillna(0).ge(min_pressure_matches)
            )
            coverage = 0.80 * pi_ok.astype(float) + 0.20 * xgb_ok.astype(float)
            weighted = (
                0.80 * candidates["pi_performance_score"].where(pi_ok, 0).fillna(0)
                + 0.20 * candidates["xgb_performance_score"].where(xgb_ok, 0).fillna(0)
            )
            candidates["model_weight_coverage"] = coverage
            candidates["decision_score"] = weighted.div(coverage.replace(0, np.nan))
            candidates = candidates.loc[coverage.gt(0)].copy()
            candidates["decision_rank"] = candidates["decision_score"].rank(method="min", ascending=False)
            candidates = candidates.sort_values(["decision_score", "phase_deliveries"], ascending=[False, False])

            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Eligible players", f"{len(candidates):,}")
            q2.metric("PI weight", "80%")
            q3.metric("XGBoost weight", "20%")
            q4.metric("Top decision score", f"{candidates['decision_score'].iloc[0]:.1f}/100" if not candidates.empty else "—")
            if candidates.empty:
                st.warning("No player meets the current per-model sample thresholds. Lower the gates or switch to the full IPL pool.")
            else:
                show = [
                    "decision_rank", "player_name", "decision_score", "model_weight_coverage",
                    "pi_performance_score", "pi_pressure_deliveries", "pi_pressure_matches",
                    "pi_mean_pressure_score", "xgb_performance_score",
                    "xgb_pressure_deliveries", "xgb_pressure_matches",
                    "xgb_mean_pressure_score", "phase_deliveries", "phase_matches",
                ]
                show = [c for c in show if c in candidates.columns]
                st.dataframe(candidates[show].rename(columns={
                    "decision_rank": "Rank", "player_name": "Player",
                    "decision_score": "PI/XGB decision score",
                    "model_weight_coverage": "Model weight coverage",
                    "pi_performance_score": "PI performance percentile",
                    "pi_pressure_deliveries": "PI pressure balls",
                    "pi_pressure_matches": "PI pressure matches",
                    "pi_mean_pressure_score": "Mean PI pressure",
                    "xgb_performance_score": "XGB performance percentile",
                    "xgb_pressure_deliveries": "XGB pressure balls",
                    "xgb_pressure_matches": "XGB pressure matches",
                    "xgb_mean_pressure_score": "Mean XGB pressure",
                    "phase_deliveries": "Total phase deliveries",
                    "phase_matches": "Total phase matches",
                }), use_container_width=True, hide_index=True)
                fig = px.bar(
                    candidates.head(15).sort_values("decision_score"), x="decision_score",
                    y="player_name", orientation="h",
                    title=f"Top {selection_role.lower()} options — {selection_phase}",
                    labels={"decision_score": "80% PI / 20% XGBoost score", "player_name": "Player"},
                )
                fig.update_layout(height=480, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
            st.info(
                "Batters are scored on strike rate, boundary rate and dismissal avoidance in high-pressure balls. "
                "Bowlers are scored on economy, wickets and dot-ball rate when the defending side is under pressure. "
                "The PI/XG scores identify the pressure deliveries; performance metrics determine who handled them best."
            )
            st.warning(
                "The XGBoost model was trained on synthetic teacher labels. Its score is a supporting signal, "
                "not validated psychological-pressure ground truth. Review PI-only and XGB-only scores and sample sizes."
            )

    with xi_tab:
        st.subheader("RR 2026 season frequent XI")
        st.caption(
            "XI membership is calculated exclusively from Rajasthan Royals playing-XI listings in IPL 2026. "
            "The PI/XG phase performance scores still use the wider 2022–2026 IPL delivery sample. "
            "This is the 2026 season XI, not a claim about the official 2027 retained squad."
        )
        if frequent_xi.empty:
            st.info("No playing-XI list was found. The pipeline falls back to observed RR batters and bowlers.")
        else:
            xi_cols = [
                "historical_xi_rank", "player_name", "appearances", "first_season", "last_season",
                "usage_profile", "rr_batting_balls", "rr_batting_runs", "rr_batting_strike_rate",
                "rr_bowling_legal_balls", "rr_bowling_wickets", "rr_bowling_economy",
            ]
            xi_cols = [c for c in xi_cols if c in frequent_xi.columns]
            st.dataframe(frequent_xi[xi_cols], use_container_width=True, hide_index=True)
            st.markdown("#### Each frequent-XI player's batting and bowling options by phase")
            if xi_options.empty:
                st.info("No phase options generated.")
            else:
                option_cols = [
                    "historical_xi_rank", "player_name", "role", "phase",
                    "decision_score", "decision_rank", "pi_performance_score",
                    "pi_pressure_deliveries", "xgb_performance_score",
                    "xgb_pressure_deliveries", "phase_option_status",
                ]
                option_cols = [c for c in option_cols if c in xi_options.columns]
                st.dataframe(xi_options[option_cols].rename(columns={
                    "historical_xi_rank": "2026 XI rank", "player_name": "Player",
                    "role": "Option", "phase": "Phase", "decision_score": "PI/XGB score",
                    "decision_rank": "Phase rank", "pi_performance_score": "PI percentile",
                    "pi_pressure_deliveries": "PI pressure balls",
                    "xgb_performance_score": "XGB percentile",
                    "xgb_pressure_deliveries": "XGB pressure balls",
                    "phase_option_status": "Sample status",
                }), use_container_width=True, hide_index=True)
                st.caption(
                    "Every player appears in all six options: batting and bowling in Powerplay, Middle and Death. "
                    "A missing score means the player has no qualifying high-pressure sample for that option."
                )




with tabs[11]:
    st.subheader("Final submission rankings + RR XI replacement plan")
    submission_tab, rr_plan_tab = st.tabs(["Final Top 10 — IPL + T20I", "RR XI weakness → replacement"])

    with submission_tab:
        st.caption(
            "Competition rankings use the supplied IPL + T20I data. Each PI variant is ranked independently, "
            "then combined through weighted Reciprocal Rank Fusion. Defaults: 50 pressure balls, 5 matches, "
            "support from 3 methods, 365-day activity window; a 100-delivery prior shrinks small samples."
        )
        rankings = data["rankings"]
        if rankings.empty:
            st.warning("Player rankings are missing. Run python run_analysis.py, then reopen the dashboard.")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                final_half_life = st.selectbox("Recency horizon", ["365d", "540d", "180d", "730d", "unweighted"], index=0, key="final_rrf_half_life")
            with c2:
                final_min_balls = st.number_input("Minimum pressure balls", 0, 10000, 50, 10, key="final_rrf_min_balls")
            with c3:
                final_min_matches = st.number_input("Minimum matches", 0, 1000, 5, 1, key="final_rrf_min_matches")
            with c4:
                final_min_support = st.number_input("Minimum model support", 1, 5, 3, 1, key="final_rrf_min_support")
            with c5:
                final_max_days = st.number_input("Days since last pressure match", 30, 2000, 365, 30, key="final_rrf_max_days")

            with st.spinner("Building final model-consensus lists..."):
                final_ensemble = build_weighted_rrf_ensemble(
                    rankings,
                    min_deliveries=int(final_min_balls),
                    min_matches=int(final_min_matches),
                    half_life=final_half_life,
                    max_days_since_last_match=int(final_max_days),
                    min_model_support=int(final_min_support),
                    method_weights=DEFAULT_RRF_WEIGHTS,
                )
            if final_ensemble.empty:
                st.error("No candidates meet the current sample and model-support gates. Lower the thresholds or change the recency horizon.")
            else:
                batters_final = final_ensemble.loc[
                    final_ensemble["role"].eq("Batter") & final_ensemble["ensemble_rank"].le(10)
                ].sort_values("ensemble_rank").copy()
                bowlers_final = final_ensemble.loc[
                    final_ensemble["role"].eq("Bowler") & final_ensemble["ensemble_rank"].le(10)
                ].sort_values("ensemble_rank").copy()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Eligible batters", f"{(final_ensemble['role'] == 'Batter').sum():,}")
                m2.metric("Eligible bowlers", f"{(final_ensemble['role'] == 'Bowler').sum():,}")
                m3.metric("PI model weight", "95%")
                m4.metric("Synthetic XGB weight", "5%")
                if "pressure_uplift_per_100" not in rankings.columns:
                    st.warning("These saved rankings predate personal-baseline uplift. Run python run_analysis.py again to regenerate the updated Top 10 lists.")

                st.markdown("**Final top 10 batters**")
                bat_cols = [
                    "ensemble_rank", "player_name", "rrf_score", "mean_selection_score",
                    "mean_pressure_uplift_per_100", "model_support", "min_pressure_deliveries_supported",
                    "min_pressure_matches_supported", "rank_pi_baseline", "rank_pi_rrr_heavy",
                    "rank_pi_momentum_heavy", "rank_pi_stakes_heavy", "rank_xgboost",
                ]
                bat_cols = [c for c in bat_cols if c in batters_final.columns]
                st.dataframe(batters_final[bat_cols].rename(columns={
                    "ensemble_rank": "Rank", "player_name": "Player", "rrf_score": "RRF score",
                    "mean_selection_score": "Model score", "mean_pressure_uplift_per_100": "Pressure uplift / 100 balls",
                    "model_support": "Models supporting", "min_pressure_deliveries_supported": "Min pressure balls",
                    "min_pressure_matches_supported": "Min pressure matches",
                }), use_container_width=True, hide_index=True)

                st.markdown("**Final top 10 bowlers**")
                bowl_cols = [c for c in bat_cols if c in bowlers_final.columns]
                st.dataframe(bowlers_final[bowl_cols].rename(columns={
                    "ensemble_rank": "Rank", "player_name": "Player", "rrf_score": "RRF score",
                    "mean_selection_score": "Model score", "mean_pressure_uplift_per_100": "Wicket-rate uplift / 100 balls",
                    "model_support": "Models supporting", "min_pressure_deliveries_supported": "Min pressure balls",
                    "min_pressure_matches_supported": "Min pressure matches",
                }), use_container_width=True, hide_index=True)

                export_cols = [c for c in [
                    "role", "ensemble_rank", "player_id", "player_name", "rrf_score",
                    "mean_selection_score", "mean_pressure_uplift_per_100", "model_support",
                    "model_weight_coverage", "min_pressure_deliveries_supported",
                    "max_pressure_deliveries_supported", "min_pressure_matches_supported",
                    "max_pressure_matches_supported", "days_since_last_pressure_match",
                    "rank_pi_baseline", "rank_pi_rrr_heavy", "rank_pi_momentum_heavy",
                    "rank_pi_stakes_heavy", "rank_xgboost",
                ] if c in final_ensemble.columns]
                final_export = pd.concat([batters_final, bowlers_final], ignore_index=True)
                st.download_button(
                    "Download combined final Top 10 + Top 10 CSV",
                    data=final_export[export_cols].to_csv(index=False).encode("utf-8"),
                    file_name=f"supeRR_final_top10_{final_half_life}.csv", mime="text/csv",
                    key="download_final_supeer_top20",
                )
                c_bat, c_bowl = st.columns(2)
                with c_bat:
                    st.download_button(
                        "Download batter Top 10", data=batters_final[export_cols].to_csv(index=False).encode("utf-8"),
                        file_name=f"supeRR_top10_batters_{final_half_life}.csv", mime="text/csv",
                        key="download_final_batters",
                    )
                with c_bowl:
                    st.download_button(
                        "Download bowler Top 10", data=bowlers_final[export_cols].to_csv(index=False).encode("utf-8"),
                        file_name=f"supeRR_top10_bowlers_{final_half_life}.csv", mime="text/csv",
                        key="download_final_bowlers",
                    )
                with st.expander("Ranking weights and interpretation"):
                    st.write(pd.DataFrame([{"Method": k, "Weight": f"{v:.0%}"} for k, v in DEFAULT_RRF_WEIGHTS.items()]))
                    st.markdown(
                        "- PI baseline receives the largest vote; RRR-, momentum- and stakes-heavy PI variants test model sensitivity. "
                        "Synthetic-trained XGBoost is a supporting 5% signal because its labels are generated from a designed teacher rule.\n"
                        "- Batters are assessed when the batting side is under pressure. Bowlers are assessed when the defending side is under pressure, using the inverse chase state.\n"
                        "- Batters combine pressure scoring rate with uplift over their own lower-pressure baseline. Bowlers combine wicket rate, economy, dot-ball rate and wicket-rate uplift.\n"
                        "- RRF requires cross-model support; sample gates and recency control candidate reliability."
                    )
                st.warning("Verify spelling and dropdown eligibility before Unstop submission. Pressure uplift is a within-player comparison, not proof of a psychological trait.")

    with rr_plan_tab:
        st.caption(
            "The XI is the most frequently listed RR playing XI in IPL 2026. Individual pressure records use RR chase deliveries "
            "from IPL 2022–2026; replacements are drawn from the wider IPL player pool. "
            "The final-over lens is the last six balls of a chase (balls_remaining ≤ 6), separate from the broader Death phase."
        )
        xi_review = data["rr_xi_review"]
        replacement_shortlist = data["rr_replacements"]
        if xi_review.empty:
            st.info("RR review outputs are not available yet. Run the following after python run_analysis.py:")
            st.code(
                "python scripts/build_rr_intelligence.py --download --start-year 2022 --end-year 2026 --xi-season 2026\n"
                "streamlit run dashboard/app.py",
                language="bash",
            )
        else:
            flagged = int(xi_review["weakness_status"].ne("No strong weakness signal").sum())
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("XI role/phase profiles", f"{len(xi_review):,}")
            q2.metric("Weakness / squad-need flags", f"{flagged:,}")
            q3.metric("Replacement options", f"{len(replacement_shortlist):,}")
            q4.metric("PI share of replacement score", "68%")

            f1, f2, f3 = st.columns(3)
            with f1:
                review_role = st.selectbox("Role", ["All", "Batter", "Bowler"], key="rr_final_review_role")
            with f2:
                review_phase = st.selectbox("Phase", ["All", "Powerplay", "Middle", "Death"], key="rr_final_review_phase")
            with f3:
                review_status = st.selectbox(
                    "Signal",
                    ["All", "Pressure weakness supported by RR sample",
                     "Squad need — individual RR pressure sample is limited", "No strong weakness signal"],
                    key="rr_final_review_status",
                )
            filtered_review = xi_review.copy()
            if review_role != "All":
                filtered_review = filtered_review.loc[filtered_review["role"].eq(review_role)]
            if review_phase != "All":
                filtered_review = filtered_review.loc[filtered_review["phase"].eq(review_phase)]
            if review_status != "All":
                filtered_review = filtered_review.loc[filtered_review["weakness_status"].eq(review_status)]
            review_cols = [c for c in [
                "historical_xi_rank", "player_name", "role", "phase", "pi_type",
                "team_need_score", "decision_score", "weakness_score", "weakness_status",
                "rr_pi_pressure_deliveries", "rr_pi_pressure_matches", "pressure_sr_uplift",
                "pressure_boundary_uplift", "pressure_dismissal_delta", "pressure_economy_delta",
                "pressure_wicket_delta", "pressure_dot_delta",
                "rr_last_over_deliveries", "rr_last_over_matches",
                "rr_last_over_strike_rate", "rr_last_over_baseline_strike_rate",
                "rr_last_over_uplift", "rr_last_over_boundary_delta",
                "rr_last_over_dismissal_delta", "rr_last_over_wicket_delta",
                "rr_last_over_dot_delta", "rr_last_over_economy", "rr_last_over_baseline_economy",
                "rr_last_over_wickets_per_100", "rr_last_over_dot_rate",
                "rr_last_over_sample_adequate", "last_over_weakness_score", "weakness_reason",
            ] if c in filtered_review.columns]
            st.dataframe(filtered_review[review_cols].rename(columns={
                "historical_xi_rank": "2026 XI rank", "player_name": "Player", "pi_type": "Selected PI",
                "team_need_score": "RR phase need", "decision_score": "All-IPL PI/XGB score",
                "weakness_score": "Weakness score", "weakness_status": "Finding",
                "rr_pi_pressure_deliveries": "RR PI pressure balls", "rr_pi_pressure_matches": "RR PI pressure matches",
                "pressure_sr_uplift": "Pressure SR uplift", "pressure_boundary_uplift": "Boundary-rate uplift",
                "pressure_dismissal_delta": "Dismissal delta / 100", "pressure_economy_delta": "Pressure economy delta",
                "pressure_wicket_delta": "Wicket-rate delta / 100", "pressure_dot_delta": "Dot-rate delta",
                "rr_last_over_deliveries": "Last-over pressure balls",
                "rr_last_over_matches": "Last-over pressure matches",
                "rr_last_over_strike_rate": "Last-over pressure SR",
                "rr_last_over_baseline_strike_rate": "Last-over baseline SR",
                "rr_last_over_uplift": "Last-over SR / economy uplift",
                "rr_last_over_boundary_delta": "Last-over boundary-rate delta",
                "rr_last_over_dismissal_delta": "Last-over dismissal delta",
                "rr_last_over_wicket_delta": "Last-over wicket-rate delta",
                "rr_last_over_dot_delta": "Last-over dot-rate delta",
                "rr_last_over_economy": "Last-over pressure economy",
                "rr_last_over_baseline_economy": "Last-over baseline economy",
                "rr_last_over_wickets_per_100": "Last-over wickets / 100",
                "rr_last_over_dot_rate": "Last-over dot rate",
                "rr_last_over_sample_adequate": "Last-over sample adequate",
                "last_over_weakness_score": "Last-over weakness score",
                "weakness_reason": "Evidence",
            }), use_container_width=True, hide_index=True)

            st.markdown("#### Phase-matched replacement shortlist")
            if replacement_shortlist.empty:
                st.info("No candidate cleared the available role/phase pressure sample gates.")
            else:
                replacements_for = sorted(replacement_shortlist["replaces_player"].dropna().unique().tolist())
                selected_weak_player = st.selectbox("Replace which XI player?", replacements_for, key="rr_replacement_target")
                repl = replacement_shortlist.loc[
                    replacement_shortlist["replaces_player"].eq(selected_weak_player)
                ].sort_values(["weak_role", "weak_phase", "replacement_rank"])
                repl_cols = [c for c in [
                    "weak_role", "weak_phase", "replacement_rank", "player_name", "candidate_score",
                    "decision_score", "pi_type", "pi_performance_score", "xgb_performance_score",
                    "pi_pressure_deliveries", "pi_pressure_matches", "xgb_pressure_deliveries",
                    "player_fit_score", "franchises", "candidate_sample_status",
                    "candidate_fit_reason", "availability_status",
                    "last_over_fit_score", "last_over_deliveries", "last_over_matches",
                    "last_over_sample_adequate", "last_over_candidate_evidence",
                ] if c in repl.columns]
                st.dataframe(repl[repl_cols].rename(columns={
                    "weak_role": "Role", "weak_phase": "Phase", "replacement_rank": "Replacement rank",
                    "player_name": "Candidate", "candidate_score": "Candidate score",
                    "decision_score": "80/20 PI/XGB score", "pi_type": "Selected PI",
                    "pi_performance_score": "PI performance", "xgb_performance_score": "XGB performance",
                    "pi_pressure_deliveries": "PI pressure balls", "pi_pressure_matches": "PI pressure matches",
                    "xgb_pressure_deliveries": "XGB pressure balls", "player_fit_score": "Role/phase fit",
                    "franchises": "Observed franchises", "candidate_sample_status": "Sample gate",
                    "candidate_fit_reason": "Fit rationale", "availability_status": "Availability",
                    "last_over_fit_score": "Final-over fit score",
                    "last_over_deliveries": "Final-over pressure balls",
                    "last_over_matches": "Final-over matches",
                    "last_over_sample_adequate": "Final-over sample adequate",
                    "last_over_candidate_evidence": "Final-over evidence",
                }), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download RR replacement shortlist",
                    data=replacement_shortlist.to_csv(index=False).encode("utf-8"),
                    file_name="rr_xi_replacement_shortlist.csv", mime="text/csv",
                    key="download_rr_replacements",
                )
            st.info(
                "Replacement score = 85% phase PI/XGBoost decision score + 15% player-fit score. The phase model is 80% PI / 20% synthetic XGBoost. "
                "Powerplay uses baseline PI, Middle uses momentum-heavy PI, and Death uses RRR-situation-heavy PI. "
                "For Death-over replacements, final-over pressure performance can add 15% to the score only when "
                "the candidate has at least 8 final-over pressure balls across 2 matches; otherwise the phase score is used. "
                "Candidates exclude the current XI and normally need at least 20 PI-pressure deliveries over 3 matches; relaxed gates are labelled."
            )
            st.warning(
                "The frequent XI is historical IPL 2026 usage, not an official 2027 retained squad. "
                "Auction availability, overseas slots, budget and retention status are not inferred."
            )


st.divider()
st.markdown(
    "<p class='muted'>Methodology: PI is a transparent heuristic; synthetic XGBoost learns a declared synthetic rule; the historical outcome XGBoost predicts actual chase outcomes on a later-match holdout. These are three different purposes. Unresolved bye/legbye values and the lack of tournament standings remain data limitations.</p>",
    unsafe_allow_html=True,
)
