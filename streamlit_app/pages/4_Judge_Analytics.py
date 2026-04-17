# streamlit_app/pages/4_Judge_Analytics.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

st.set_page_config(page_title="Judge Analytics", page_icon="👨‍⚖️", layout="wide")
st.title("👨‍⚖️ Judge Pattern Analytics")
st.caption("Explore ruling patterns, specializations, and activity across judges.")

@st.cache_data
def load_judge_stats():
    path = "models/judge_stats.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

stats = load_judge_stats()

if not stats:
    st.warning("Judge stats not found. Run `python src/ml/judge_analysis.py` first.")
    st.stop()

# ── Overall stats ────────────────────────────────────────────────
st.subheader("Overview")
judges_list = list(stats.keys())
total_judges = len(judges_list)
st.metric("Total judges in dataset", total_judges)

# Build summary DataFrame
rows = []
for judge, s in stats.items():
    rows.append({
        "Judge":          judge,
        "Total Cases":    s["total_cases"],
        "Allowed":        s["outcome_counts"].get("allowed", 0),
        "Dismissed":      s["outcome_counts"].get("dismissed", 0),
        "Allowed Rate":   s["allowed_rate"],
        "Dismissed Rate": s["dismissed_rate"],
        "Top Keywords":   ", ".join(s["top_keywords"][:3]),
    })
df = pd.DataFrame(rows)

# ── Top judges by caseload ───────────────────────────────────────
st.subheader("Top 15 judges by caseload")
top15 = df.nlargest(15, "Total Cases")

fig = go.Figure()
fig.add_trace(go.Bar(
    name="Allowed",
    x=top15["Judge"].str[:25],
    y=top15["Allowed"],
    marker_color="#1D9E75",
))
fig.add_trace(go.Bar(
    name="Dismissed",
    x=top15["Judge"].str[:25],
    y=top15["Dismissed"],
    marker_color="#E24B4A",
))
fig.update_layout(
    barmode="stack",
    height=420,
    xaxis_tickangle=-35,
    legend=dict(orientation="h", y=1.08),
    margin=dict(l=20, r=20, t=40, b=120),
)
st.plotly_chart(fig, use_container_width=True)

# ── Allowed rate scatter ─────────────────────────────────────────
st.subheader("Allowed rate vs caseload")
st.caption("Judges who handled more cases and their tendency to allow vs dismiss.")

fig2 = px.scatter(
    df[df["Total Cases"] >= 5],
    x="Total Cases",
    y="Allowed Rate",
    hover_name="Judge",
    color="Allowed Rate",
    color_continuous_scale=["#E24B4A","#EF9F27","#1D9E75"],
    size="Total Cases",
    size_max=30,
)
fig2.update_layout(height=420, margin=dict(l=20,r=20,t=20,b=20))
st.plotly_chart(fig2, use_container_width=True)

# ── Judge selector ───────────────────────────────────────────────
st.markdown("---")
st.subheader("Judge deep dive")

selected_judge = st.selectbox("Select a judge:", judges_list[:100])

if selected_judge:
    s = stats[selected_judge]

    d1, d2, d3 = st.columns(3)
    d1.metric("Total cases",   s["total_cases"])
    d2.metric("Allowed rate",  f"{s['allowed_rate']:.1%}")
    d3.metric("Dismissed rate",f"{s['dismissed_rate']:.1%}")

    c1, c2 = st.columns(2)
    with c1:
        oc = s["outcome_counts"]
        fig3 = go.Figure(go.Pie(
            labels=list(oc.keys()),
            values=list(oc.values()),
            marker_colors=["#1D9E75","#E24B4A","#EF9F27","#378ADD","#888780"],
            hole=0.4,
        ))
        fig3.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        st.markdown("**Top legal areas**")
        for kw in s.get("top_keywords", []):
            st.markdown(f"- `{kw}`")
        yrs = s.get("years_active", [])
        if yrs:
            st.markdown(f"**Active years:** {yrs[0]} – {yrs[-1]}")

    st.dataframe(
        df[df["Judge"] == selected_judge],
        use_container_width=True,
        hide_index=True,
    )