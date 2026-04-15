# src/ml/train_classifier.py
"""
OUTCOME PREDICTION CLASSIFIER:
Trains a simple but effective ML classifier to predict
whether a case will be: allowed / dismissed / partly allowed / remanded
Features: TF-IDF on case text + metadata features
Logs everything to MLflow
"""
import dagshub
dagshub.init(repo_owner='pranjal79', repo_name='legal-case-search', mlflow=True)
import pandas as pd
import numpy as np
import pickle
import os
import json
import mlflow
import mlflow.sklearn
import yaml
import logging
from sklearn.pipeline          import Pipeline
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing     import LabelEncoder
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import (classification_report,
                                       accuracy_score,
                                       f1_score,
                                       confusion_matrix)
from sklearn.compose           import ColumnTransformer
from sklearn.preprocessing     import OneHotEncoder
from scipy.sparse              import hstack

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def prepare_features(df: pd.DataFrame):
    """
    Build feature matrix from:
    - TF-IDF on case_facts + legal_keywords (text features)
    - Year as numeric feature
    - Court as categorical feature
    """
    df = df.copy()
    df["text_features"] = (
        df["case_facts"].fillna("") + " " +
        df["legal_keywords"].fillna("") + " " +
        df["petitioner"].fillna("") + " " +
        df["respondent"].fillna("")
    )
    df["year_num"] = pd.to_numeric(df["year"], errors="coerce").fillna(1990)
    return df


def build_label_encoder(df: pd.DataFrame):
    """Encode outcome labels, keeping only main 4 classes."""
    valid_outcomes = ["allowed", "dismissed", "partly allowed", "remanded"]
    df = df[df["outcome"].isin(valid_outcomes)].copy()
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["outcome"])
    logger.info(f"Label distribution:\n{df['outcome'].value_counts()}")
    return df, le


def train_and_evaluate(config):
    clean_path = config["data"]["processed_path"]
    model_dir  = "models"
    os.makedirs(model_dir, exist_ok=True)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # ── Load and prepare data ─────────────────────────────────────
    logger.info(f"Loading data from {clean_path}")
    df = pd.read_csv(clean_path).fillna("")
    df = prepare_features(df)
    df, le = build_label_encoder(df)

    logger.info(f"Training on {len(df)} labeled cases")

    # ── Features ─────────────────────────────────────────────────
    X_text = df["text_features"]
    y      = df["label"]

    X_train_t, X_test_t, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Models to compare ────────────────────────────────────────
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=500, C=1.0, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
    }

    best_model_name = None
    best_f1         = 0
    best_pipeline   = None

    for model_name, clf in models.items():
        logger.info(f"\nTraining: {model_name}")

        with mlflow.start_run(run_name=f"classifier_{model_name}"):

            # Build pipeline: TF-IDF → Classifier
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),   # unigrams + bigrams
                    sublinear_tf=True,    # log scaling
                    stop_words="english",
                )),
                ("clf", clf),
            ])

            # Train
            pipe.fit(X_train_t, y_train)

            # Evaluate
            y_pred  = pipe.predict(X_test_t)
            acc     = accuracy_score(y_test, y_pred)
            f1_mac  = f1_score(y_test, y_pred, average="macro")
            f1_wt   = f1_score(y_test, y_pred, average="weighted")
            cv_scores = cross_val_score(pipe, X_train_t, y_train, cv=3, scoring="f1_macro")

            # Log to MLflow
            mlflow.log_param("model_type",    model_name)
            mlflow.log_param("max_features",  10000)
            mlflow.log_param("ngram_range",   "(1,2)")
            mlflow.log_param("n_train",       len(X_train_t))
            mlflow.log_param("n_test",        len(X_test_t))

            mlflow.log_metric("accuracy",     round(acc, 4))
            mlflow.log_metric("f1_macro",     round(f1_mac, 4))
            mlflow.log_metric("f1_weighted",  round(f1_wt, 4))
            mlflow.log_metric("cv_f1_mean",   round(cv_scores.mean(), 4))
            mlflow.log_metric("cv_f1_std",    round(cv_scores.std(), 4))

            report = classification_report(
                y_test, y_pred, target_names=le.classes_, output_dict=True
            )
            mlflow.log_dict(report, "classification_report.json")

            # Log the sklearn model
            mlflow.sklearn.log_model(pipe, artifact_path=f"model_{model_name}")

            logger.info(f"  Accuracy : {acc:.3f}")
            logger.info(f"  F1 Macro : {f1_mac:.3f}")
            logger.info(f"  CV F1    : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(classification_report(y_test, y_pred, target_names=le.classes_))

            if f1_mac > best_f1:
                best_f1         = f1_mac
                best_model_name = model_name
                best_pipeline   = pipe

    # ── Save best model ──────────────────────────────────────────
    logger.info(f"\n🏆 Best model: {best_model_name} (F1={best_f1:.3f})")

    best_path = os.path.join(model_dir, "outcome_classifier.pkl")
    le_path   = os.path.join(model_dir, "label_encoder.pkl")

    with open(best_path, "wb") as f:
        pickle.dump(best_pipeline, f)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    # Save metrics summary
    metrics = {
        "best_model": best_model_name,
        "best_f1_macro": round(best_f1, 4),
        "n_classes": len(le.classes_),
        "classes": list(le.classes_),
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"✅ Best model saved to {best_path}")
    return best_pipeline, le


if __name__ == "__main__":
    config = load_config()
    train_and_evaluate(config)