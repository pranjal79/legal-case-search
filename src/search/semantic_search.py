# src/search/semantic_search.py
"""
SEMANTIC SEARCH ENGINE:
Loads FAISS index + MongoDB
Given a natural language query → returns top-K most similar cases
"""

import faiss
import pickle
import numpy as np
import os
import yaml
import logging
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LegalCaseSearchEngine:
    """
    Main search engine class.
    Load once, call search() many times — no reloading on every query.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r" , encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self._load_model()
        self._load_faiss()
        self._load_mongodb()
        logger.info("✅ Search engine ready!")

    def _load_model(self):
        model_name = self.config["model"]["embedding_model"]
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def _load_faiss(self):
        emb_dir    = self.config["data"]["embeddings_path"]
        faiss_path = os.path.join(emb_dir, "faiss_index.bin")
        ids_path   = os.path.join(emb_dir, "case_ids.pkl")

        logger.info(f"Loading FAISS index from {faiss_path}")
        self.index = faiss.read_index(faiss_path)

        with open(ids_path, "rb") as f:
            self.case_ids = pickle.load(f)

        logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")

    def _load_mongodb(self):
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        client = MongoClient(uri)
        db = client["legal_cases_db"]
        self.collection = db["cases"]
        logger.info("MongoDB connected")

    def _embed_query(self, query: str) -> np.ndarray:
        """Convert query string to normalized embedding vector."""
        vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return vec.astype(np.float32)

    def search(self, query: str, top_k: int = 5, filters: dict = None):
        """
        Main search method.

        Args:
            query:   Natural language search query
            top_k:   Number of results to return
            filters: Optional dict for MongoDB filtering
                     e.g. {"year": "1980", "outcome": "allowed"}

        Returns:
            List of case dicts with similarity scores
        """
        # Step 1: Embed the query
        query_vec = self._embed_query(query)

        # Step 2: Search FAISS — get top_k * 3 to allow for filtering
        search_k = top_k * 3
        scores, indices = self.index.search(query_vec, search_k)

        # Step 3: Fetch matching cases from MongoDB
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:          # FAISS returns -1 for unfilled slots
                continue

            case_id = self.case_ids[idx]

            # Build MongoDB query
            mongo_query = {"case_id": case_id}
            if filters:
                mongo_query.update(filters)

            case = self.collection.find_one(
                mongo_query,
                {"_id": 0}        # exclude MongoDB internal _id field
            )

            if case:
                case["similarity_score"] = round(float(score), 4)
                results.append(case)

            if len(results) >= top_k:
                break

        return results

    def get_case_by_id(self, case_id: str):
        """Fetch a single case by ID from MongoDB."""
        return self.collection.find_one({"case_id": case_id}, {"_id": 0})

    def get_similar_to_case(self, case_id: str, top_k: int = 5):
        """Find cases similar to a given case (by its ID)."""
        case = self.get_case_by_id(case_id)
        if not case:
            return []

        # Build query from case content
        query = f"{case.get('case_title','')} {case.get('legal_keywords','')} {case.get('case_facts','')[:200]}"
        results = self.search(query, top_k=top_k + 1)

        # Exclude the case itself from results
        return [r for r in results if r["case_id"] != case_id][:top_k]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = LegalCaseSearchEngine()

    test_queries = [
        "fundamental rights violation by state",
        "property dispute land acquisition",
        "criminal appeal murder conviction",
        "constitutional validity of law",
        "habeas corpus detention illegal",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 60)
        results = engine.search(query, top_k=3)
        for i, case in enumerate(results, 1):
            print(f"  {i}. [{case['similarity_score']:.3f}] {case['case_title'][:70]}")
            print(f"      Year: {case['year']} | Outcome: {case.get('outcome','?')}")