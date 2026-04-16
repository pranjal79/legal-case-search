# src/utils/mlflow_logger.py
"""
Centralized MLflow logger.
Import this in ANY script to get consistent experiment tracking.
Usage:
    from src.utils.mlflow_logger import MLflowLogger
    logger = MLflowLogger("my_run_name")
    logger.log_params({...})
    logger.log_metrics({...})
    logger.end_run()
"""

import mlflow
import mlflow.sklearn
import os
import yaml
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_mlflow(config: dict = None):
    """
    Configure MLflow to log to DAGsHub.
    Call this once at the top of any script before creating runs.
    """
    if config is None:
        config = load_config()

    tracking_uri  = config["mlflow"]["tracking_uri"]
    experiment    = config["mlflow"]["experiment_name"]

    # Set DAGsHub credentials from environment
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv(
        "MLFLOW_TRACKING_USERNAME",
        os.getenv("DAGSHUB_USERNAME", "")
    )
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv(
        "MLFLOW_TRACKING_PASSWORD",
        os.getenv("DAGSHUB_TOKEN", "")
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    log.info(f"MLflow tracking URI : {tracking_uri}")
    log.info(f"MLflow experiment   : {experiment}")


class MLflowLogger:
    """
    Context-manager-style wrapper around MLflow runs.

    Usage (as context manager — recommended):
        with MLflowLogger("train_classifier") as ml:
            ml.log_params({"model": "lr", "C": 1.0})
            ml.log_metrics({"accuracy": 0.82})
            ml.log_model(sklearn_model, "classifier")

    Usage (manual):
        ml = MLflowLogger("embed_cases")
        ml.log_params({...})
        ml.log_metrics({...})
        ml.end_run()
    """

    def __init__(self, run_name: str, config: dict = None, tags: dict = None):
        self.config   = config or load_config()
        self.run_name = run_name
        self.tags     = tags or {}
        self._run     = None
        setup_mlflow(self.config)

    def __enter__(self):
        self._run = mlflow.start_run(
            run_name=self.run_name,
            tags={
                "project": "legal-case-search",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                **self.tags,
            }
        )
        log.info(f"MLflow run started: {self.run_name} | ID: {self._run.info.run_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error", str(exc_val))
        else:
            mlflow.set_tag("status", "SUCCESS")
        mlflow.end_run()
        log.info(f"MLflow run ended: {self.run_name}")

    def log_params(self, params: dict):
        """Log a dict of hyperparameters / config values."""
        for k, v in params.items():
            mlflow.log_param(k, v)

    def log_metrics(self, metrics: dict, step: int = None):
        """Log a dict of numeric metrics."""
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v), step=step)

    def log_model(self, model, artifact_path: str, flavor: str = "sklearn"):
        """Log a trained model artifact."""
        if flavor == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path=artifact_path)
        else:
            mlflow.pyfunc.log_model(artifact_path, python_model=model)
        log.info(f"Model logged: {artifact_path}")

    def log_artifact(self, local_path: str):
        """Log a local file as an MLflow artifact."""
        mlflow.log_artifact(local_path)

    def log_dict(self, data: dict, filename: str):
        """Log a dict as a JSON artifact."""
        mlflow.log_dict(data, filename)

    def set_tag(self, key: str, value: str):
        mlflow.set_tag(key, value)

    def end_run(self):
        mlflow.end_run()