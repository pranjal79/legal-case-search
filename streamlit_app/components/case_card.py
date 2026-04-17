# streamlit_app/components/case_card.py
import streamlit as st

OUTCOME_COLORS = {
    "allowed":        "green",
    "dismissed":      "red",
    "partly allowed": "orange",
    "remanded":       "blue",
    "unknown":        "gray",
}

def outcome_badge(outcome: str) -> str:
    outcome = (outcome or "unknown")
    color = OUTCOME_COLORS.get(outcome.lower(), "gray")
    return f":{color}[**{outcome.upper()}**]"


def safe_text(value, max_len=40):
    """Safely handle None values and slicing."""
    if value is None:
        return "N/A"
    return str(value)[:max_len]


def render_case_card(case: dict, show_score: bool = True, key_prefix: str = ""):
    """
    Renders a single case as a Streamlit card with expandable details.
    """

    outcome  = case.get("outcome") or "unknown"
    score    = case.get("similarity_score")
    title    = safe_text(case.get("case_title"), 80)
    year     = case.get("year") or "?"
    court    = case.get("court") or "Unknown Court"
    keywords = case.get("legal_keywords") or ""
    case_id  = case.get("case_id") or ""

    # Preview text (safe)
    raw_text = case.get("text_preview") or case.get("judgment_text_clean") or ""
    preview  = safe_text(raw_text, 350)

    with st.container():

        col1, col2 = st.columns([5, 1])

        # LEFT: Title + court
        with col1:
            st.markdown(f"### {title}")
            st.caption(f"{court} · {year}")

        # RIGHT: Outcome + score
        with col2:
            st.markdown(outcome_badge(outcome))
            if show_score and score is not None:
                st.caption(f"Score: {score:.3f}")

        # Keywords
        if keywords:
            kw_list = [k.strip() for k in str(keywords).split(",")[:6] if k.strip()]
            if kw_list:
                st.markdown(" ".join([f"`{k}`" for k in kw_list]))

        # Preview
        st.markdown(f"*{preview}...*")

        # Parties + Judges (SAFE NOW)
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.caption(f"Petitioner: {safe_text(case.get('petitioner'))}")

        with col_b:
            st.caption(f"Respondent: {safe_text(case.get('respondent'))}")

        with col_c:
            st.caption(f"Judge(s): {safe_text(case.get('judges'))}")

        # Full text expander
        with st.expander("View full judgment text"):
            full_text = case.get("judgment_text_clean") or "No text available."
            st.text_area(
                "Judgment",
                str(full_text)[:3000],
                height=300,
                key=f"{key_prefix}_{case_id}_text",
                disabled=True,
            )

        # Citations
        citations = case.get("citations")
        if citations and str(citations).lower() != "nan":
            with st.expander("Citations"):
                for c in str(citations).split("|"):
                    if c.strip():
                        st.markdown(f"- {c.strip()}")