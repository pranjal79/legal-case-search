# src/ml/citation_graph.py

"""
CITATION GRAPH:
Builds a directed graph of case citations.
Node  = one legal case
Edge  = case A cites case B
Used for: finding influential cases, tracing legal precedents
"""

import pandas as pd
import networkx as nx
import pickle
import os
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CitationGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._case_metadata = {}

    def build_from_dataframe(self, df: pd.DataFrame):
        logger.info("Building citation graph...")

        df["year"] = df["year"].astype(str)

        # ── STEP 1: Add nodes ─────────────────────────────
        for _, row in df.iterrows():
            case_id = str(row.get("case_id", "")).strip()
            if not case_id:
                continue

            self.graph.add_node(case_id)

            self._case_metadata[case_id] = {
                "title":   row.get("case_title", ""),
                "year":    row.get("year", ""),
                "court":   row.get("court", ""),
                "outcome": row.get("outcome", ""),
            }

        # ── STEP 2: Build YEAR MAP ────────────────────────
        year_map = {}
        for _, row in df.iterrows():
            year = str(row.get("year", "")).strip()
            case_id = str(row.get("case_id", "")).strip()

            year_map.setdefault(year, []).append({
                "case_id": case_id,
                "title": str(row.get("case_title", "")).lower()
            })

        # ── STEP 3: Add edges ────────────────────────────
        edges_added = 0

        for _, row in df.iterrows():
            source_id = str(row.get("case_id", "")).strip()
            citations = str(row.get("citations", "")).strip()

            if not source_id or not citations or citations.lower() == "nan":
                continue

            cited_refs = [c.strip() for c in citations.split("|") if len(c.strip()) > 5]

            for ref in cited_refs:
                target = self._find_case_by_citation(ref, year_map)

                if target and target != source_id:
                    self.graph.add_edge(source_id, target)
                    edges_added += 1

        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {edges_added} edges")
        return self

    def _find_case_by_citation(self, ref: str, year_map: dict) -> str:
        ref = ref.lower().strip()

        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', ref)
        if not year_match:
            return None

        year = year_match.group()

        if year not in year_map:
            return None

        candidates = year_map[year]

        ref_words = [w for w in ref.split() if len(w) > 3]

        for case in candidates:
            if any(word in case["title"] for word in ref_words[:5]):
                return case["case_id"]

        return candidates[0]["case_id"] if candidates else None

    def get_most_cited(self, top_n: int = 10) -> list:
        in_degrees = dict(self.graph.in_degree())
        sorted_cases = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)

        results = []
        for case_id, degree in sorted_cases[:top_n]:
            meta = self._case_metadata.get(case_id, {})
            results.append({
                "case_id": case_id,
                "title": meta.get("title", case_id),
                "year": meta.get("year", ""),
                "citations_received": degree,
            })
        return results

    def compute_pagerank(self, top_n: int = 10) -> list:
        if self.graph.number_of_edges() == 0:
            logger.warning("No edges in graph — PageRank skipped")
            return []

        pr = nx.pagerank(self.graph, alpha=0.85)
        sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)

        results = []
        for case_id, score in sorted_pr[:top_n]:
            meta = self._case_metadata.get(case_id, {})
            results.append({
                "case_id": case_id,
                "title": meta.get("title", case_id),
                "year": meta.get("year", ""),
                "pagerank": round(score, 6),
            })
        return results

    # ✅ ADD THIS METHOD (FIXES YOUR ERROR)
    def get_subgraph_for_viz(self, case_id, depth=2):
        if case_id not in self.graph:
            return {"nodes": [], "edges": []}

        nodes = set([case_id])
        edges = []
        frontier = [case_id]

        for _ in range(depth):
            next_frontier = []

            for node in frontier:
                # outgoing edges
                for nbr in self.graph.successors(node):
                    edges.append({"source": node, "target": nbr})
                    if nbr not in nodes:
                        nodes.add(nbr)
                        next_frontier.append(nbr)

                # incoming edges
                for nbr in self.graph.predecessors(node):
                    edges.append({"source": nbr, "target": node})
                    if nbr not in nodes:
                        nodes.add(nbr)
                        next_frontier.append(nbr)

            frontier = next_frontier

        node_data = []
        for n in nodes:
            meta = self._case_metadata.get(n, {})

            node_data.append({
                "id": n,
                "label": str(meta.get("title", n))[:40],
                "year": meta.get("year", "?"),
                "outcome": meta.get("outcome", "unknown"),
                "is_root": n == case_id
            })

        return {
            "nodes": node_data,
            "edges": edges
        }

    def save(self, path: str = "models/citation_graph.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Graph saved to {path}")

    @classmethod
    def load(cls, path: str = "models/citation_graph.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)


def build_and_save_graph(clean_path: str):
    df = pd.read_csv(clean_path).fillna("")

    graph = CitationGraph()
    graph.build_from_dataframe(df)

    print(f"\n📊 Citation Graph Stats:")
    print(f"   Nodes (cases)  : {graph.graph.number_of_nodes()}")
    print(f"   Edges (cites)  : {graph.graph.number_of_edges()}")

    top_cited = graph.get_most_cited(5)
    print(f"\n🏆 Most cited cases:")
    for c in top_cited:
        print(f"   [{c['citations_received']} citations] {c['title'][:60]}")

    top_pr = graph.compute_pagerank(5)
    if top_pr:
        print(f"\n🔗 Most influential by PageRank:")
        for c in top_pr:
            print(f"   [{c['pagerank']:.5f}] {c['title'][:60]}")

    graph.save("models/citation_graph.pkl")
    return graph


if __name__ == "__main__":
    build_and_save_graph("data/processed/cases_clean.csv")