from __future__ import annotations
import json
import sys
from pathlib import Path

from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cricket_pressure.pressure import DEFAULT_WEIGHTS
from cricket_pressure.synthetic import FEATURE_COLUMNS, CLASS_NAMES, CLASS_MIDPOINTS

st.set_page_config(page_title="SupeRR Pressure Lab", page_icon="🏏", layout="wide")
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px;}
    [data-testid="stMetric"] {background: #111827; border: 1px solid #263244; padding: 14px 16px; border-radius: 12px;}
    [data-testid="stMetricLabel"] {color: #9ca3af;}
    [data-testid="stMetricValue"] {color: #f9fafb;}
    .hero {padding: 20px 22px; border: 1px solid #263244; border-radius: 16px; background: linear-gradient(110deg,#111827,#18283a); margin-bottom: 20px;}
    .hero h1 {margin: 0 0 6px 0; font-size: 2rem;}
    .hero p {color: #a9b6c7; margin: 0;}
    .muted {color: #94a3b8; font-size: .9rem;}
</style>
<div class="hero"><h1>🏏 SupeRR Pressure Lab</h1><p>Compare the transparent PI heuristic against an XGBoost model trained on explicitly synthetic pressure scenarios.</p></div>
""", unsafe_allow_html=True)

OUT = ROOT / "outputs"
PROC = ROOT / "data" / "processed" / "real_pressure_comparison.csv.gz"
MODEL_PATH = ROOT / "models" / "synthetic_pressure_xgb.json"
SUMMARY_PATH = OUT / "metrics" / "run_summary.json"
METRICS_PATH = OUT / "metrics" / "experiment_metrics.json"

if not PROC.exists() or not MODEL_PATH.exists():
    st.warning("Analysis outputs are not available yet. Run the notebook or `python run_analysis.py` from the project root first.")
    st.code("pip install -r requirements.txt\npython run_analysis.py\nstreamlit run dashboard/app.py", language="bash")
    st.stop()

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(PROC, low_memory=False)
    players = pd.read_csv(OUT / "player_pressure_comparison.csv") if (OUT / "player_pressure_comparison.csv").exists() else pd.DataFrame()
    by_season = pd.read_csv(OUT / "match_format_season_summary.csv") if (OUT / "match_format_season_summary.csv").exists() else pd.DataFrame()
    sensitivity = pd.read_csv(OUT / "weight_sensitivity.csv") if (OUT / "weight_sensitivity.csv").exists() else pd.DataFrame()
    scenario = pd.read_csv(OUT / "scenario_lab.csv") if (OUT / "scenario_lab.csv").exists() else pd.DataFrame()
    importance = pd.read_csv(OUT / "xgb_feature_importance.csv") if (OUT / "xgb_feature_importance.csv").exists() else pd.DataFrame()
    return df, players, by_season, sensitivity, scenario, importance

@st.cache_resource(show_spinner=False)
def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model

@st.cache_data(show_spinner=False)
def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

df, players, by_season, sensitivity, scenario_lab, importance = load_data()
model = load_model()
summary = load_json(str(SUMMARY_PATH)) if SUMMARY_PATH.exists() else {}
metrics = load_json(str(METRICS_PATH)) if METRICS_PATH.exists() else {}

with st.sidebar:
    st.header("Explore data")
    formats = sorted(df["match_format"].dropna().astype(str).unique().tolist())
    selected_formats = st.multiselect("Format", formats, default=formats)
    seasons = sorted(df["season"].dropna().astype(str).unique().tolist())
    selected_seasons = st.multiselect("Season", seasons, default=seasons)
    st.divider()
    st.caption("PI weights used in the baseline")
    for key, value in DEFAULT_WEIGHTS.items():
        st.write(f"{key.replace('_', ' ').title()}: **{value:.1%}**")
    st.divider()
    st.caption("Synthetic XGBoost is a rule-learning experiment, not a model trained on real pressure labels or historical winners.")

view = df.loc[df["match_format"].astype(str).isin(selected_formats) & df["season"].astype(str).isin(selected_seasons)].copy()
if view.empty:
    st.info("No rows match the selected filters.")
    st.stop()

st.markdown("### Overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Chase deliveries", f"{len(view):,}")
c2.metric("Matches", f"{view['match_id'].nunique():,}")
c3.metric("Mean PI", f"{view['pressure_index'].mean():.1f}")
c4.metric("Mean XGB score", f"{view['xgb_pressure_score'].mean():.1f}")
c5.metric("PI ↔ XGB Spearman", f"{view[['pressure_index','xgb_pressure_score']].corr(method='spearman').iloc[0,1]:.3f}")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Pressure scores across real match states")
    sample = view.sample(min(5000, len(view)), random_state=11)
    fig = px.scatter(sample, x="pressure_index", y="xgb_pressure_score", color="match_format",
                     hover_data=["match_id", "season", "required_run_rate", "wickets_remaining", "balls_remaining"],
                     opacity=.55, labels={"pressure_index": "PI heuristic", "xgb_pressure_score": "Synthetic-trained XGBoost"})
    fig.update_layout(height=430, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Pressure class mix")
    cat = pd.concat([
        view["pressure_category"].astype(str).value_counts().rename("PI"),
        view["xgb_pressure_class"].astype(str).value_counts().rename("XGBoost"),
    ], axis=1).reindex(CLASS_NAMES).fillna(0).reset_index(names="Pressure class")
    fig = px.bar(cat, x="Pressure class", y=["PI", "XGBoost"], barmode="group", height=430)
    fig.update_layout(margin=dict(l=10,r=10,t=20,b=10), legend_title="Model")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["Synthetic model diagnostics", "Scenario playground", "Weight lab", "Player comparison"])

with tab1:
    st.subheader("XGBoost — synthetic holdout evaluation")
    st.caption("Metrics below answer: how well did XGBoost learn the synthetic teacher rule? They do not answer whether the teacher rule is the true cricket-pressure definition.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Synthetic accuracy", f"{metrics.get('accuracy', float('nan')):.3f}")
    m2.metric("Macro F1", f"{metrics.get('macro_f1', float('nan')):.3f}")
    m3.metric("Synthetic score MAE", f"{metrics.get('teacher_score_mae_vs_class_midpoint', float('nan')):.2f}")
    m4.metric("Synthetic test scenarios", f"{metrics.get('n_test', 0):,}")
    l, r = st.columns(2)
    with l:
        imp = importance.sort_values("importance", ascending=True)
        fig = px.bar(imp, x="importance", y="feature", orientation="h", title="Feature importance")
        fig.update_layout(height=390, margin=dict(l=10,r=10,t=45,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with r:
        cm = np.asarray(metrics.get("confusion_matrix", np.zeros((4,4))))
        fig = px.imshow(cm, x=CLASS_NAMES, y=CLASS_NAMES, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted synthetic class", y="Synthetic teacher class", color="Scenarios"),
                        title="Synthetic holdout confusion matrix")
        fig.update_layout(height=390, margin=dict(l=10,r=10,t=45,b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.warning("Synthetic labels are generated from an explicit nonlinear rule plus noise. High test performance demonstrates learnability of that rule, not independent validation of real pressure.")

with tab2:
    st.subheader("Build a match situation")
    st.caption("Change the situation and compare the hand-weighted PI against the synthetic-trained XGBoost output.")
    a, b, c = st.columns(3)
    with a:
        rrr = st.slider("Required run rate", 0.5, 30.0, 13.0, 0.5)
        wickets_left = st.slider("Wickets remaining", 0, 10, 5)
    with b:
        overs_total = st.selectbox("Scheduled overs", [10, 15, 20], index=2)
        overs_left = st.slider("Overs remaining", 0.5, float(overs_total), min(10.0, float(overs_total)), 0.5)
    with c:
        recent_rr = st.slider("Recent run rate", 0.0, 24.0, 9.0, 0.5)
        target_pct = st.slider("Target difficulty percentile", 0.0, 1.0, 0.70, 0.05)
        stakes = st.slider("Match stakes (0–1)", 0.0, 1.0, 0.25, 0.05)
    balls_left = max(1, int(round(overs_left * 6)))
    scheduled_balls = int(overs_total * 6)
    scenario = pd.DataFrame([{
        "required_run_rate": rrr, "wickets_remaining": wickets_left,
        "balls_remaining": balls_left, "scheduled_balls": scheduled_balls,
        "recent_run_rate_12": recent_rr, "target_difficulty": target_pct,
        "match_stakes": stakes, "death_over_pressure": float(np.clip(1 - balls_left/30,0,1)),
        "is_t20i": 0,
    }])
    scenario["rrr_pressure"] = np.clip(rrr/12,0,1)
    scenario["wicket_pressure"] = np.clip(1-wickets_left/10,0,1)
    scenario["time_urgency"] = np.clip(1-balls_left/scheduled_balls,0,1)
    scenario["momentum_pressure"] = np.clip(max(rrr-recent_rr,0)/12,0,1)
    scenario["target_difficulty"] = target_pct
    scenario["match_stakes"] = stakes
    pi_score = sum(scenario[k].iloc[0] * w for k,w in DEFAULT_WEIGHTS.items()) * 100
    prob = model.predict_proba(scenario[FEATURE_COLUMNS])[0]
    xgb_score = float(prob @ CLASS_MIDPOINTS)
    pred_cls = CLASS_NAMES[int(prob.argmax())]
    z1,z2,z3 = st.columns(3)
    z1.metric("PI heuristic", f"{pi_score:.1f}/100")
    z2.metric("XGBoost expected score", f"{xgb_score:.1f}/100")
    z3.metric("XGBoost predicted class", pred_cls)
    prob_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": prob})
    fig = px.bar(prob_df, x="Class", y="Probability", range_y=[0,1], title="XGBoost synthetic class probabilities")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Death-over contribution: {scenario['death_over_pressure'].iloc[0]:.2f} · Balls remaining: {balls_left}")

with tab3:
    st.subheader("PI weight sensitivity lab")
    st.caption("The sliders change the heuristic weights. We normalize the selected values to sum to 100% before calculating scores.")
    keys = list(DEFAULT_WEIGHTS)
    cols = st.columns(4)
    chosen = {}
    for i, key in enumerate(keys):
        with cols[i % 4]:
            chosen[key] = st.slider(key.replace("_", " ").title(), 0.0, 0.50, float(DEFAULT_WEIGHTS[key]), 0.01, key=f"weight_{key}")
    total = sum(chosen.values())
    if total <= 0:
        st.error("At least one weight must be above zero.")
    else:
        norm = {k:v/total for k,v in chosen.items()}
        score = sum(view[k].fillna(0) * w for k,w in norm.items()) * 100
        rho = float(spearmanr(view["pressure_index"], score).statistic)
        st.write("Normalized weights", {k: f"{v:.1%}" for k,v in norm.items()})
        q1,q2,q3 = st.columns(3)
        q1.metric("Adjusted PI mean", f"{score.mean():.1f}")
        q2.metric("Spearman vs baseline PI", f"{rho:.3f}")
        q3.metric("Mean absolute score change", f"{(score-view['pressure_index']).abs().mean():.2f}")
        fig = px.histogram(pd.DataFrame({"Baseline PI":view["pressure_index"],"Adjusted PI":score}), barmode="overlay", opacity=.6,
                           title="Baseline versus adjusted PI distribution")
        st.plotly_chart(fig, use_container_width=True)
    if not sensitivity.empty:
        st.subheader("Pre-computed weight scenarios")
        st.dataframe(sensitivity, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Player pressure exposure")
    st.caption("These are exposure and descriptive performance summaries, not final player rankings. Bowler conceded runs remain provisional because of unresolved byes/legbyes.")
    if players.empty:
        st.info("Player summaries were not generated.")
    else:
        roles = sorted(players["role"].dropna().unique().tolist())
        role = st.selectbox("Role", roles)
        p = players.loc[players["role"].eq(role)].copy()
        sort_col = "runs_per_100_pressure_deliveries" if role == "Batter" else "wickets_per_100_pressure_deliveries"
        if sort_col in p:
            p = p.sort_values(sort_col, ascending=False)
        st.dataframe(p.head(100), use_container_width=True, hide_index=True)

st.divider()
st.markdown("<p class='muted'>Methodology note: the PI is a transparent heuristic. XGBoost is trained only on synthetic labels generated by a declared rule. PI–XGBoost agreement on historical rows measures model agreement, not predictive validity. The supplied data has unresolved bye/legbye scoring fields and no tournament standings table to establish must-win status.</p>", unsafe_allow_html=True)