# src/ml/generate_embeddings.py

"""
EMBEDDING STEP:
Loads cleaned cases CSV
Generates embeddings using sentence-transformers
Builds FAISS index
Logs metrics to MLflow (local by default)
"""
import dagshub
dagshub.init(repo_owner='pranjal79', repo_name='legal-case-search', mlflow=True)
import pandas as pd
import numpy as np
import faiss
import pickle
import os
import time
import mlflow
import yaml
import logging
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config():
    with open("configs/config.yaml", "r" , encoding="utf-8" ) as f:
        return yaml.safe_load(f)


def build_search_text(row) -> str:
    parts = [
        str(row.get("case_title", "")),
        str(row.get("case_facts", ""))[:500],
        str(row.get("legal_keywords", "")),
        str(row.get("outcome", "")),
        str(row.get("court", "")),
        str(row.get("year", "")),
    ]
    return " ".join([p for p in parts if p and p != "nan"])


def generate_embeddings(df: pd.DataFrame, model_name: str, batch_size: int = 64):
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info("Building search texts...")
    texts = df.apply(build_search_text, axis=1).tolist()   # faster than iterrows

    logger.info(f"Generating embeddings for {len(texts)} cases...")
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    elapsed = time.time() - start
    logger.info(f"Embeddings generated in {elapsed:.1f}s")
    logger.info(f"Embedding shape: {embeddings.shape}")

    return embeddings, elapsed


def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    logger.info(f"Building FAISS index (dim={dim})")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    logger.info(f"FAISS index built. Total vectors: {index.ntotal}")
    return index


def save_artifacts(index, case_ids, embeddings_dir):
    os.makedirs(embeddings_dir, exist_ok=True)

    faiss_path = os.path.join(embeddings_dir, "faiss_index.bin")
    ids_path = os.path.join(embeddings_dir, "case_ids.pkl")

    faiss.write_index(index, faiss_path)

    with open(ids_path, "wb") as f:
        pickle.dump(case_ids, f)

    logger.info(f"Saved FAISS index → {faiss_path}")
    logger.info(f"Saved case IDs → {ids_path}")

    return faiss_path, ids_path


def setup_mlflow(config):
    """Safe MLflow setup with fallback to local."""
    try:
        tracking_uri = config["mlflow"].get("tracking_uri", "file:./mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
    except Exception:
        logger.warning("⚠️ Failed to set tracking URI. Using local MLflow.")
        mlflow.set_tracking_uri("file:./mlruns")

    try:
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
    except Exception:
        logger.warning("⚠️ Failed to set experiment. Using default.")


def run(config):
    clean_path = config["data"]["processed_path"]
    embeddings_dir = config["data"]["embeddings_path"]
    model_name = config["model"]["embedding_model"]
    batch_size = 64

    # ✅ FIXED MLflow setup
    setup_mlflow(config)

    with mlflow.start_run(run_name="generate_embeddings"):

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("normalize", True)
        mlflow.log_param("index_type", "IndexFlatIP")

        logger.info(f"Loading data from {clean_path}")
        df = pd.read_csv(clean_path, encoding="utf-8", encoding_errors="ignore").fillna("")
        mlflow.log_param("n_cases", len(df))

        embeddings, elapsed = generate_embeddings(df, model_name, batch_size)

        mlflow.log_metric("embedding_time_sec", round(elapsed, 2))
        mlflow.log_metric("n_vectors", embeddings.shape[0])
        mlflow.log_metric("embedding_dim", embeddings.shape[1])

        index = build_faiss_index(embeddings)

        case_ids = df["case_id"].tolist()
        faiss_path, ids_path = save_artifacts(index, case_ids, embeddings_dir)

        mlflow.log_artifact(faiss_path)
        mlflow.log_artifact(ids_path)

        logger.info("✅ Embedding generation complete!")

        print("\n📊 Summary:")
        print(f"   Cases embedded : {embeddings.shape[0]}")
        print(f"   Embedding dim  : {embeddings.shape[1]}")
        print(f"   Time taken     : {elapsed:.1f}s")
        print(f"   FAISS vectors  : {index.ntotal}")


if __name__ == "__main__":
    config = load_config()
    run(config)