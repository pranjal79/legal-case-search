# streamlit_app/components/sidebar.py
import streamlit as st

def render_sidebar():
    """
    Renders the shared sidebar with filters.
    Returns a dict of selected filter values.
    """
    st.sidebar.title("⚖️ Legal Case Search")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Filters")

    year_options = ["All"] + [str(y) for y in range(1950, 2024)]
    year = st.sidebar.selectbox("Year", year_options, index=0)

    outcome = st.sidebar.selectbox(
        "Outcome",
        ["All", "Allowed", "Dismissed", "Partly Allowed", "Remanded", "Unknown"]
    )

    top_k = st.sidebar.slider("Results to show", min_value=3, max_value=20, value=5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "Searches 26,000+ Supreme Court judgments "
        "using semantic similarity — not just keywords."
    )

    return {
        "year":    year,
        "outcome": outcome.lower() if outcome != "All" else None,
        "top_k":   top_k,
    }