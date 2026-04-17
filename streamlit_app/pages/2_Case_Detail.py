# streamlit_app/pages/2_Case_Detail.py
import streamlit as st
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from streamlit_app.components.case_card import render_case_card, outcome_badge

st.set_page_config(page_title="Case Detail", page_icon="📄", layout="wide")
st.title("📄 Case Detail View")


# ✅ SAFE HELPER (IMPORTANT)
def safe_text(value, max_len=None):
    if value is None:
        return "N/A"
    value = str(value)
    return value[:max_len] if max_len else value


engine     = st.session_state.get("engine")
predictor  = st.session_state.get("predictor")
summarizer = st.session_state.get("summarizer")

# ── Case selector ────────────────────────────────────────────────
last_results = st.session_state.get("last_results", [])

if last_results:
    options = {
        f"{safe_text(c.get('case_title'),60)} ({c.get('year','?')})": c
        for c in last_results
    }
    selected_label = st.selectbox("Select a case from your last search:", list(options.keys()))
    case = options[selected_label]
else:
    case_id_input = st.text_input("Or enter a Case ID directly:", placeholder="1950_A_K_Gopalan_vs_...")
    if case_id_input and engine:
        case = engine.get_case_by_id(case_id_input.strip())
        if not case:
            st.error("Case not found.")
            st.stop()
    else:
        st.info("Run a search first, or enter a case ID above.")
        st.stop()

if not case:
    st.stop()

# ── Case header ──────────────────────────────────────────────────
st.markdown(f"## {safe_text(case.get('case_title'))}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Year",    safe_text(case.get("year")))
m2.metric("Court",   safe_text(case.get("court"), 25))
m3.metric("Outcome", safe_text(case.get("outcome"), 25).title())
m4.metric("Judge(s)", safe_text(case.get("judges"), 25))

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Prediction", "Similar Cases", "Full Text"])

# TAB 1: Auto Summary
with tab1:
    st.subheader("AI-Generated Summary")
    text = case.get("judgment_text_clean") or ""

    if st.button("Generate Summary", type="primary"):
        if summarizer and text:
            with st.spinner("Summarizing judgment..."):
                summary = summarizer.summarize(text)
            st.success("Summary generated!")
            st.markdown(f"> {summary}")
            st.session_state[f"summary_{case.get('case_id')}"] = summary
        else:
            st.warning("Summarizer not available or no text found.")

    cached = st.session_state.get(f"summary_{case.get('case_id')}")
    if cached:
        st.markdown(f"> {cached}")

    st.markdown("---")
    st.subheader("Case Facts (first 500 chars)")
    st.markdown(safe_text(case.get("case_facts"), 500))

    st.subheader("Legal Keywords")
    kws_raw = case.get("legal_keywords") or ""
    kws = [k.strip() for k in str(kws_raw).split(",") if k.strip()]
    if kws:
        st.markdown(" ".join([f"`{k}`" for k in kws]))
    else:
        st.caption("No keywords extracted.")

    citations = case.get("citations")
    if citations and str(citations).lower() != "nan":
        st.subheader("Citations in this judgment")
        for c in str(citations).split("|"):
            if c.strip():
                st.markdown(f"- {c.strip()}")

# TAB 2: Outcome Prediction
with tab2:
    st.subheader("Outcome Prediction")
    st.caption("ML model predicts the likely outcome based on case text and keywords.")

    if st.button("Predict Outcome", type="primary"):
        if predictor:
            with st.spinner("Running prediction..."):
                result = predictor.predict(
                    case.get("judgment_text_clean") or "",
                    case.get("legal_keywords") or "",
                )

            pred     = result["predicted_outcome"]
            conf     = result["confidence"]
            conf_lbl = result["confidence_label"]
            all_prob = result["all_probabilities"]

            st.markdown(f"### Predicted: {outcome_badge(pred)}")
            st.progress(conf, text=f"{conf:.1%} — {conf_lbl}")

            st.subheader("All class probabilities")
            import plotly.graph_objects as go

            fig = go.Figure(go.Bar(
                x=list(all_prob.keys()),
                y=list(all_prob.values()),
                marker_color=["#1D9E75","#E24B4A","#EF9F27","#378ADD"],
            ))

            fig.update_layout(
                yaxis_title="Probability",
                xaxis_title="Outcome",
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
            )

            st.plotly_chart(fig, use_container_width=True)

            actual = case.get("outcome") or "unknown"
            if actual != "unknown":
                if actual.lower() == pred.lower():
                    st.success(f"Actual outcome: **{actual.upper()}** — prediction matches!")
                else:
                    st.warning(f"Actual outcome: **{actual.upper()}** — prediction differs.")
        else:
            st.warning("Predictor not loaded.")

# TAB 3: Similar Cases
with tab3:
    st.subheader("Similar Precedent Cases")
    if engine:
        with st.spinner("Finding similar cases..."):
            similar = engine.get_similar_to_case(case.get("case_id") or "", top_k=5)

        if similar:
            for i, sim_case in enumerate(similar):
                render_case_card(sim_case, show_score=True, key_prefix=f"sim_{i}")
        else:
            st.info("No similar cases found.")

# TAB 4: Full Text
with tab4:
    st.subheader("Full Judgment Text")
    full_text = case.get("judgment_text_clean") or "No text available."

    st.text_area("", str(full_text), height=500, disabled=True)

    st.download_button(
        "Download judgment text",
        data=str(full_text),
        file_name=f"{case.get('case_id','case')}.txt",
        mime="text/plain",
    )