# streamlit_app/app.py
"""
Main entry point for the Legal Case Search Streamlit app.
Initializes shared resources once using st.session_state
so models are not reloaded on every page interaction.
"""

import streamlit as st
import sys
import os

# Make sure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(
    page_title="Legal Case Search",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading search engine...")
def load_search_engine():
    from src.search.semantic_search import LegalCaseSearchEngine
    return LegalCaseSearchEngine()


@st.cache_resource(show_spinner="Loading predictor...")
def load_predictor():
    from src.ml.predictor import OutcomePredictor
    return OutcomePredictor()


@st.cache_resource(show_spinner="Loading summarizer...")
def load_summarizer():
    from src.ml.summarizer import CaseSummarizer
    return CaseSummarizer()


@st.cache_resource(show_spinner="Loading citation graph...")
def load_graph():
    from src.ml.citation_graph import CitationGraph
    try:
        return CitationGraph.load("models/citation_graph.pkl")
    except Exception:
        return None


# Load into session state so all pages can access them
if "engine" not in st.session_state:
    st.session_state.engine     = load_search_engine()
    st.session_state.predictor  = load_predictor()
    st.session_state.summarizer = load_summarizer()
    st.session_state.graph      = load_graph()

# ── Home page ────────────────────────────────────────────────────
st.title("⚖️ Legal Case Law Search & Precedent Finder")
st.markdown("""
Welcome to the **Supreme Court of India Case Search System**.
Use the pages in the sidebar to:
""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("**Search**\nFind similar cases using natural language")
with col2:
    st.info("**Case Detail**\nView summary, prediction & citations")
with col3:
    st.info("**Citation Graph**\nExplore the network of precedents")
with col4:
    st.info("**Judge Analytics**\nAnalyze judge patterns & statistics")

st.markdown("---")
st.markdown("*Powered by sentence-transformers · FAISS · MongoDB · HuggingFace · MLflow*")