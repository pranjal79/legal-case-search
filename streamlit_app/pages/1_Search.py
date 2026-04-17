# streamlit_app/pages/1_Search.py
import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.case_card import render_case_card
from src.search.search_utils import parse_query_filters

st.set_page_config(page_title="Search Cases", page_icon="🔍", layout="wide")

filters = render_sidebar()

st.title("🔍 Semantic Case Search")
st.caption("Search using natural language — not just keywords.")

# ── Search bar ───────────────────────────────────────────────────
query = st.text_input(
    "Search query",
    placeholder="e.g. 'fundamental rights violated by state detention' or 'property acquisition compensation'",
    label_visibility="collapsed",
)

col_search, col_clear = st.columns([1, 5])
with col_search:
    search_clicked = st.button("Search", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear"):
        query = ""

# ── Example queries ──────────────────────────────────────────────
st.markdown("**Try these examples:**")
examples = [
    "fundamental rights article 19 freedom of speech",
    "land acquisition compensation property owner",
    "criminal appeal murder conviction IPC 302",
    "writ of habeas corpus illegal detention",
    "contract breach damages civil liability",
]
cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    with cols[i]:
        if st.button(ex[:35] + "...", key=f"ex_{i}", use_container_width=True):
            query = ex
            search_clicked = True

# ── Run search ───────────────────────────────────────────────────
if (search_clicked or query) and query.strip():
    engine = st.session_state.get("engine")
    if engine is None:
        st.error("Search engine not loaded. Please restart the app.")
        st.stop()

    mongo_filters = parse_query_filters(
        year    = filters["year"],
        outcome = filters["outcome"],
    )

    with st.spinner(f"Searching for '{query}'..."):
        results = engine.search(
            query,
            top_k   = filters["top_k"],
            filters = mongo_filters,
        )

    if not results:
        st.warning("No results found. Try a different query or remove filters.")
    else:
        st.success(f"Found **{len(results)}** similar cases")

        # Save results to session for use on other pages
        st.session_state["last_results"] = results
        st.session_state["last_query"]   = query

        for i, case in enumerate(results):
            render_case_card(case, show_score=True, key_prefix=f"search_{i}")

elif not query and search_clicked:
    st.warning("Please enter a search query.")