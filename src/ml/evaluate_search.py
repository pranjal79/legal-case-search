# src/ml/evaluate_search.py
"""
Evaluate search quality and log results to MLflow.
Uses a small set of known queries with expected keywords in results.
"""

import mlflow
import yaml
import time
from src.search.semantic_search import LegalCaseSearchEngine


TEST_QUERIES = [
    {"query": "fundamental rights article 19",    "expect_keyword": "fundamental"},
    {"query": "property acquisition compensation", "expect_keyword": "property"},
    {"query": "criminal appeal high court",        "expect_keyword": "criminal"},
    {"query": "writ petition habeas corpus",       "expect_keyword": "habeas"},
    {"query": "contract breach damages",           "expect_keyword": "contract"},
]


def evaluate(config):
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    engine = LegalCaseSearchEngine()

    with mlflow.start_run(run_name="search_evaluation"):
        mlflow.log_param("n_test_queries", len(TEST_QUERIES))
        mlflow.log_param("top_k", 5)

        total_time = 0
        hits = 0

        for item in TEST_QUERIES:
            start = time.time()
            results = engine.search(item["query"], top_k=5)
            elapsed = time.time() - start
            total_time += elapsed

            # Check if expected keyword appears in any result title/keywords
            hit = any(
                item["expect_keyword"] in (r.get("legal_keywords","") + r.get("case_title","")).lower()
                for r in results
            )
            if hit:
                hits += 1

            print(f"Query: '{item['query']}' → {len(results)} results in {elapsed:.2f}s | Hit: {hit}")

        precision_at_5 = hits / len(TEST_QUERIES)
        avg_latency    = total_time / len(TEST_QUERIES)

        mlflow.log_metric("precision_at_5", round(precision_at_5, 3))
        mlflow.log_metric("avg_latency_sec", round(avg_latency, 3))
        mlflow.log_metric("total_hits", hits)

        print(f"\n📊 Evaluation Results:")
        print(f"   Precision@5   : {precision_at_5:.1%}")
        print(f"   Avg latency   : {avg_latency*1000:.0f}ms")


if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    evaluate(config)