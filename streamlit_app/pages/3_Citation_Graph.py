# streamlit_app/pages/3_Citation_Graph.py

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import pickle
import sys, os

# ✅ IMPORTANT: Add this import (fixes pickle error)
from src.ml.citation_graph import CitationGraph

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

st.set_page_config(page_title="Citation Graph", page_icon="🕸️", layout="wide")
st.title("🕸️ Case Citation Network")
st.caption("Explore how Supreme Court cases cite each other.")


# ✅ SAFE TEXT HELPER
def safe_text(value, max_len=None):
    if value is None:
        return "N/A"
    value = str(value)
    return value[:max_len] if max_len else value


# ✅ LOAD GRAPH FROM FILE
@st.cache_resource
def load_graph():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    path = os.path.join(BASE_DIR, "models", "citation_graph.pkl")

    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        graph = pickle.load(f)

    return graph


# 🔥 LOAD GRAPH
graph_obj = load_graph()

if graph_obj is None:
    st.error("❌ Citation graph not found.")
    st.info("👉 Run this in terminal:\n\n    python src/ml/citation_graph.py")
    st.stop()


G = graph_obj.graph

# ── Global stats ─────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total cases (nodes)", f"{G.number_of_nodes():,}")
c2.metric("Citation links (edges)", f"{G.number_of_edges():,}")

# Most cited
top_cited_1 = graph_obj.get_most_cited(1)
if top_cited_1:
    title_cited = safe_text(top_cited_1[0].get("title"), 30)
    c3.metric("Most cited case", title_cited + "...")
else:
    c3.metric("Most cited case", "N/A")

# PageRank
top_pr = graph_obj.compute_pagerank(1)
if top_pr:
    title_pr = safe_text(top_pr[0].get("title"), 30)
    c4.metric("Top PageRank case", title_pr + "...")
else:
    c4.metric("Top PageRank case", "N/A")

st.markdown("---")

tab1, tab2 = st.tabs(["Most Cited Cases", "Case Subgraph Explorer"])


# ── TAB 1 ────────────────────────────────────────────────────────
with tab1:
    st.subheader("Top 10 most cited cases")
    top_cited = graph_obj.get_most_cited(10)

    if top_cited:
        import pandas as pd

        df_cited = pd.DataFrame(top_cited)

        df_cited["title"] = df_cited["title"].apply(lambda x: safe_text(x, 50))
        df_cited["year"] = df_cited["year"].apply(safe_text)

        df_cited = df_cited[["title", "year", "citations_received"]]
        df_cited.columns = ["Case Title", "Year", "Times Cited"]

        fig = go.Figure(go.Bar(
            x=df_cited["Times Cited"],
            y=df_cited["Case Title"],
            orientation="h",
        ))

        fig.update_layout(
            height=420,
            xaxis_title="Times Cited",
            yaxis=dict(autorange="reversed"),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_cited, use_container_width=True)

        # PageRank table
        top_pr_list = graph_obj.compute_pagerank(10)
        if top_pr_list:
            st.subheader("Top 10 by PageRank")

            df_pr = pd.DataFrame(top_pr_list)
            df_pr["title"] = df_pr["title"].apply(lambda x: safe_text(x, 50))
            df_pr["year"] = df_pr["year"].apply(safe_text)

            df_pr = df_pr[["title", "year", "pagerank"]]
            df_pr.columns = ["Case Title", "Year", "PageRank Score"]

            st.dataframe(df_pr, use_container_width=True)

    else:
        st.info("No citation data available.")


# ── TAB 2 ────────────────────────────────────────────────────────
with tab2:
    st.subheader("Explore citations around a case")

    last_results = st.session_state.get("last_results", [])

    if last_results:
        options = {
            f"{safe_text(c.get('case_title'),55)} ({c.get('year','?')})": c.get("case_id")
            for c in last_results
        }
        selected = st.selectbox("Pick a case:", list(options.keys()))
        selected_id = options[selected]
    else:
        selected_id = st.text_input("Enter case ID:")

    depth = st.slider("Citation depth", 1, 3, 2)

    if selected_id and st.button("Draw subgraph"):

        subgraph_data = graph_obj.get_subgraph_for_viz(selected_id, depth=depth)
        nodes = subgraph_data.get("nodes", [])
        edges = subgraph_data.get("edges", [])

        if len(nodes) <= 1:
            st.info("No citation links found.")
        else:
            sub_g = nx.DiGraph()

            for n in nodes:
                sub_g.add_node(n["id"])

            for e in edges:
                sub_g.add_edge(e["source"], e["target"])

            pos = nx.spring_layout(sub_g, seed=42)

            edge_x, edge_y = [], []
            for u, v in sub_g.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            node_x = [pos[n["id"]][0] for n in nodes]
            node_y = [pos[n["id"]][1] for n in nodes]

            node_colors = [
                "#534AB7" if n.get("is_root") else (
                    "#1D9E75" if n.get("outcome") == "allowed" else
                    "#E24B4A" if n.get("outcome") == "dismissed" else
                    "#888780"
                )
                for n in nodes
            ]

            node_labels = [safe_text(n.get("label"), 40) for n in nodes]
            node_years  = [safe_text(n.get("year")) for n in nodes]

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                hoverinfo="none"
            ))

            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                text=node_labels,
                textposition="top center",
                customdata=node_years,
                hovertemplate="<b>%{text}</b><br>Year: %{customdata}<extra></extra>"
            ))

            fig.update_layout(
                height=500,
                showlegend=False,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
            )

            st.plotly_chart(fig, use_container_width=True)