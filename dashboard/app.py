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
from cricket_pressure.final_selection import build_final_selection_candidates, build_consensus_rankings

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
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()

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

@st.cache_data(show_spinner="Scoring match-to-match pressure consistency...")
def load_final_selection(
    frame: pd.DataFrame, min_deliveries: int, min_matches: int, risk_penalty: float
):
    return build_final_selection_candidates(
        frame,
        min_deliveries=min_deliveries,
        min_matches=min_matches,
        risk_penalty=risk_penalty,
    )

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
    "Match disagreement", "Player rankings", "Historical win model", "Score audit", "Recency analysis", "Final Selection Lab",
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
        half_lives = ["unweighted", "365d", "540d", "730d"]
        selected_half_life = st.selectbox(
            "Recency setting",
            half_lives,
            index=2,
            format_func=lambda x: {"unweighted":"No decay", "365d":"365-day half-life",
                                  "540d":"540-day half-life", "730d":"730-day half-life"}[x],
        )
        p = rankings.loc[
            rankings["role"].eq(role)
            & rankings["scoring_method"].isin(selected_methods)
            & rankings["half_life"].eq(selected_half_life)
        ].copy()
        p["eligible_current"] = p["pressure_deliveries"].ge(min_deliveries) & p["pressure_matches"].ge(min_matches)
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
        st.dataframe(top[["scoring_method","role","current_rank","player_name","pressure_deliveries",
                          "pressure_matches","observed_per_100","shrunk_per_100","eligible_current"]],
                     use_container_width=True, hide_index=True)
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
    st.caption("A 365-day half-life halves the influence of a match one year old relative to the newest match in this dataset. The reference date is fixed to the latest valid match date in the dataset, so reruns are reproducible.")
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
        st.caption("Each half-life uses the same high-pressure threshold and raw delivery/match eligibility gates. Only the contribution of older performances changes.")
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
            for half in ["unweighted","365d","540d","730d"]:
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
                for half in ["unweighted","365d","540d","730d"]
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
    for half in [365,540,730]:
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
    st.subheader("Final Selection Lab")
    st.caption(
        "Ranks players on repeatable performance in high-pressure chase deliveries. "
        "It does not rank players by improvement over their own baseline."
    )

    control_1, control_2, control_3, control_4 = st.columns(4)
    with control_1:
        final_min_deliveries = st.number_input(
            "Minimum pressure deliveries", min_value=25, max_value=5000,
            value=100, step=25, key="final_min_deliveries",
        )
    with control_2:
        final_min_matches = st.number_input(
            "Minimum pressure matches", min_value=3, max_value=1000,
            value=10, step=1, key="final_min_matches",
        )
    with control_3:
        final_half_life = st.selectbox(
            "Recency setting",
            ["unweighted", "365d", "540d", "730d"], index=2,
            format_func=lambda x: {
                "unweighted": "No decay", "365d": "365-day half-life",
                "540d": "540-day half-life", "730d": "730-day half-life",
            }[x],
            key="final_half_life",
        )
    with control_4:
        risk_penalty = st.slider(
            "Consistency penalty", min_value=0.0, max_value=2.0,
            value=1.0, step=0.25, key="final_risk_penalty",
            help="Higher values penalize match-to-match volatility more strongly.",
        )

    with st.expander("How the consistency ranking works", expanded=False):
        st.markdown(
            """
            - **Pressure sample:** deliveries with the selected scoring method at or above 60.
            - **Batter primary metric:** runs per 100 balls faced (strike-rate scale); wides are excluded
              from balls faced. Dismissals per 100 balls faced are shown as a secondary indicator.
            - **Bowler primary metric:** dot-ball percentage on known legal deliveries. Rows with
              unresolved bye/legbye run values are excluded from the dot-ball denominator. Credited
              wickets per 100 legal balls are shown separately.
            - **Match consistency:** calculate each player's pressure performance separately in each
              match, then calculate a recency- and exposure-weighted mean and standard deviation.
            - **Reliability adjustment:** shrink the mean toward the role-wide pressure rate using a
              five-effective-match prior, then subtract the selected number of standard errors.
              Strong repeatable performances score well; volatile results and low effective samples
              are penalized.
            - **Consensus:** the final order is the median percentile across four PI definitions
              (baseline, RRR-heavy, momentum-heavy and stakes-heavy). At least three must support a
              candidate. Synthetic-trained XGBoost is shown only as a diagnostic comparison.
            
            Half-lives are anchored to the latest valid match date in the dataset. The main sidebar's
            format and season filters define the pool being compared.
            """
        )

    selection = load_final_selection(
        view, int(final_min_deliveries), int(final_min_matches), float(risk_penalty)
    )
    if selection.empty:
        st.warning(
            "No high-pressure player samples are available under the current format and season filters. "
            "Check the processed data or widen the filters."
        )
    else:
        consensus = build_consensus_rankings(selection, min_methods=3)
        if consensus.empty:
            st.warning(
                "No players qualify under at least three PI definitions and the selected sample gates. "
                "Do not lower the thresholds just to force a Top 10."
            )
        else:
            selected_consensus = consensus.loc[consensus["half_life"].eq(final_half_life)].copy()
            batter_consensus = selected_consensus.loc[selected_consensus["role"].eq("Batter")]
            bowler_consensus = selected_consensus.loc[selected_consensus["role"].eq("Bowler")]
            metric_1, metric_2, metric_3, metric_4 = st.columns(4)
            metric_1.metric("Eligible batter candidates", f"{len(batter_consensus):,}")
            metric_2.metric("Eligible bowler candidates", f"{len(bowler_consensus):,}")
            metric_3.metric("Batter Top 10", f"{min(10, len(batter_consensus))}/10")
            metric_4.metric("Bowler Top 10", f"{min(10, len(bowler_consensus))}/10")

            def _selection_columns(role_name: str):
                role_rows = selected_consensus.loc[
                    selected_consensus["role"].eq(role_name)
                ].sort_values("consensus_rank")
                top20 = role_rows.head(20)
                top10 = role_rows.head(10)

                st.markdown(f"#### Final ordered Top 10 — {role_name}s")
                if top10.empty:
                    st.info(f"No eligible {role_name.lower()} candidates under the current sample gates.")
                else:
                    display_cols = [
                        "consensus_rank", "player_name",
                        "baseline_pi_consistency_score", "baseline_pi_mean_match_rate",
                        "baseline_pi_std_match_rate", "baseline_pi_consistency_pct",
                        "baseline_pi_pressure_deliveries", "baseline_pi_pressure_matches",
                        "baseline_pi_effective_matches", "baseline_pi_secondary_rate",
                        "methods_supported", "explanation",
                    ]
                    st.dataframe(
                        top10[display_cols].rename(columns={
                            "consensus_rank": "Rank",
                            "player_name": "Player",
                            "baseline_pi_consistency_score": "Consistency score",
                            "baseline_pi_mean_match_rate": "Mean pressure-match rate",
                            "baseline_pi_std_match_rate": "Match-rate standard deviation",
                            "baseline_pi_consistency_pct": "Share of matches at/above pool rate",
                            "baseline_pi_pressure_deliveries": "Pressure deliveries",
                            "baseline_pi_pressure_matches": "Pressure matches",
                            "baseline_pi_effective_matches": "Effective matches",
                            "baseline_pi_secondary_rate": "Secondary rate",
                            "methods_supported": "PI methods supporting",
                        }),
                        use_container_width=True, hide_index=True,
                    )

                st.markdown(f"#### Top 20 {role_name.lower()} candidates")
                if top20.empty:
                    st.info("No candidates available.")
                else:
                    st.dataframe(
                        top20[[
                            "consensus_rank", "player_name", "consensus_percentile",
                            "baseline_pi_consistency_score", "baseline_pi_mean_match_rate",
                            "baseline_pi_std_match_rate", "baseline_pi_consistency_pct",
                            "baseline_pi_pressure_deliveries", "baseline_pi_pressure_matches",
                            "baseline_pi_effective_matches", "methods_supported",
                        ]].rename(columns={
                            "consensus_rank": "Consensus rank",
                            "player_name": "Player",
                            "consensus_percentile": "Median PI percentile",
                            "baseline_pi_consistency_score": "Consistency score",
                            "baseline_pi_mean_match_rate": "Mean pressure-match rate",
                            "baseline_pi_std_match_rate": "Match-rate SD",
                            "baseline_pi_consistency_pct": "Share of matches at/above pool rate",
                            "baseline_pi_pressure_deliveries": "Pressure deliveries",
                            "baseline_pi_pressure_matches": "Pressure matches",
                            "baseline_pi_effective_matches": "Effective matches",
                            "methods_supported": "PI methods supporting",
                        }),
                        use_container_width=True, hide_index=True,
                    )

                st.markdown(f"#### Top-10 and Top-20 overlap — {role_name.lower()}s")
                method_names = [
                    name for name in ["PI baseline", "PI RRR-heavy", "PI momentum-heavy",
                                      "PI stakes-heavy", "XGBoost"]
                    if name in set(selection["scoring_method"])
                ]
                for n in [10, 20]:
                    top_sets = {}
                    for method_name in method_names:
                        method_rows = selection.loc[
                            selection["role"].eq(role_name)
                            & selection["half_life"].eq(final_half_life)
                            & selection["scoring_method"].eq(method_name)
                            & selection["eligible"]
                        ].sort_values("consistency_score", ascending=False)
                        top_sets[method_name] = set(method_rows.head(n)["player_id"])
                    if method_names:
                        overlap = pd.DataFrame(
                            [[len(top_sets[a] & top_sets[b]) for b in method_names]
                             for a in method_names],
                            index=method_names, columns=method_names,
                        )
                        fig = px.imshow(
                            overlap, text_auto=True, aspect="auto", zmin=0, zmax=n,
                            color_continuous_scale="Blues",
                            title=f"Top-{n} overlap across scoring methods",
                            labels={"x": "Scoring method", "y": "Scoring method", "color": "Shared players"},
                        )
                        fig.update_layout(height=350, margin=dict(l=10, r=10, t=55, b=10))
                        st.plotly_chart(fig, use_container_width=True)

            _selection_columns("Batter")
            _selection_columns("Bowler")

            st.markdown("#### Search players")
            player_query = st.text_input(
                "Find a player by name",
                placeholder="Start typing a batter or bowler name...",
                key="final_player_search",
            )
            if player_query.strip():
                matches = selection.loc[
                    selection["player_name"].str.contains(
                        player_query.strip(), case=False, regex=False, na=False
                    )
                ].copy()
                if matches.empty:
                    st.info("No player names matched that search.")
                else:
                    st.caption(
                        "Search results show the player's pressure-only score, rank and sample "
                        "under every scoring method and recency setting."
                    )
                    search_consensus = consensus[[
                        "role", "half_life", "player_id", "consensus_rank", "methods_supported"
                    ]]
                    matches = matches.merge(
                        search_consensus,
                        on=["role", "half_life", "player_id"], how="left",
                        validate="many_to_one",
                    )
                    st.dataframe(
                        matches[[
                            "role", "player_name", "scoring_method", "half_life",
                            "rank", "consensus_rank", "consistency_score",
                            "mean_match_rate", "std_match_rate", "consistency_pct",
                            "pressure_deliveries", "pressure_matches", "effective_matches",
                            "secondary_rate", "eligible", "methods_supported",
                        ]].rename(columns={
                            "rank": "Method rank",
                            "consensus_rank": "Consensus rank",
                            "consistency_score": "Consistency score",
                            "mean_match_rate": "Mean match rate",
                            "std_match_rate": "Match-rate SD",
                            "consistency_pct": "Share of matches at/above pool rate",
                            "pressure_deliveries": "Pressure deliveries",
                            "pressure_matches": "Pressure matches",
                            "effective_matches": "Effective matches",
                            "secondary_rate": "Secondary rate",
                            "methods_supported": "PI methods supporting",
                        }),
                        use_container_width=True, hide_index=True,
                    )

            st.markdown("#### Rank movement across half-lives")
            stability = consensus.pivot_table(
                index=["role", "player_id", "player_name"],
                columns="half_life", values="consensus_rank", aggfunc="min",
            ).reset_index()
            display_halves = [
                h for h in ["unweighted", "365d", "540d", "730d"]
                if h in stability.columns
            ]
            stability = stability.sort_values(
                by=final_half_life if final_half_life in stability.columns else display_halves[0],
                na_position="last",
            )
            st.caption(
                "Lower ranks are better. A blank means the player did not qualify under at least three PI variants "
                "for that half-life."
            )
            st.dataframe(
                stability[["role", "player_name", *display_halves]].head(60),
                use_container_width=True, hide_index=True,
            )

            st.markdown("#### Top-10 and Top-20 overlap across recency settings")
            recency_overlap_rows = []
            for role_name in ["Batter", "Bowler"]:
                for n in [10, 20]:
                    sets = {}
                    for half in ["unweighted", "365d", "540d", "730d"]:
                        sets[half] = set(
                            consensus.loc[
                                consensus["role"].eq(role_name)
                                & consensus["half_life"].eq(half)
                                & consensus["consensus_rank"].le(n),
                                "player_id",
                            ]
                        )
                    anchor = sets.get(final_half_life, set())
                    for half, ids in sets.items():
                        recency_overlap_rows.append({
                            "role": role_name,
                            "list_size": f"Top {n}",
                            "half_life": half,
                            f"overlap_vs_{final_half_life}": len(anchor & ids),
                            "players_in_list": len(ids),
                        })
            st.dataframe(
                pd.DataFrame(recency_overlap_rows),
                use_container_width=True, hide_index=True,
            )
            st.warning(
                "The consistency score is a conservative estimate of pressure-only performance, not a measure "
                "of psychological clutch ability. Batter scoring uses runs per 100 balls faced. Bowler scoring "
                "uses dot-ball percentage on known legal balls; unresolved bye/legbye rows are excluded. "
                "Credited wickets are shown separately because wicket rates are sparse and volatile."
            )

st.divider()
st.markdown(
    "<p class='muted'>Methodology: PI is a transparent heuristic; synthetic XGBoost learns a declared synthetic rule; the historical outcome XGBoost predicts actual chase outcomes on a later-match holdout. These are three different purposes. Unresolved bye/legbye values and the lack of tournament standings remain data limitations.</p>",
    unsafe_allow_html=True,
)
