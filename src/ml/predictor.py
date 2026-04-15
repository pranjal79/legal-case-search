# src/ml/predictor.py
"""
PREDICTOR:
Loads the saved best classifier and label encoder.
Given case text → predicts outcome with confidence scores.
"""

import pickle
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OutcomePredictor:
    """
    Load once, predict many times.
    Used by both the search engine and Streamlit app.
    """

    def __init__(
        self,
        model_path: str = "models/outcome_classifier.pkl",
        le_path:    str = "models/label_encoder.pkl",
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run: python src/ml/train_classifier.py"
            )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(le_path, "rb") as f:
            self.le = pickle.load(f)

        logger.info("✅ Outcome predictor loaded")

    def predict(self, case_text: str, case_keywords: str = "") -> dict:
        """
        Predict outcome for a single case.

        Returns:
            {
              "predicted_outcome": "allowed",
              "confidence": 0.82,
              "all_probabilities": {"allowed": 0.82, "dismissed": 0.13, ...}
            }
        """
        combined = f"{case_text} {case_keywords}"
        combined = combined[:3000]      # safe truncation

        probs  = self.model.predict_proba([combined])[0]
        labels = self.le.classes_

        pred_idx  = int(np.argmax(probs))
        predicted = labels[pred_idx]
        confidence = float(probs[pred_idx])

        all_probs = {
            label: round(float(prob), 4)
            for label, prob in zip(labels, probs)
        }

        return {
            "predicted_outcome":  predicted,
            "confidence":         round(confidence, 4),
            "all_probabilities":  all_probs,
            "confidence_label":   self._confidence_label(confidence),
        }

    def predict_batch(self, texts: list) -> list:
        """Predict outcomes for a list of texts."""
        return [self.predict(t) for t in texts]

    @staticmethod
    def _confidence_label(confidence: float) -> str:
        if confidence >= 0.80:
            return "High confidence"
        elif confidence >= 0.60:
            return "Moderate confidence"
        else:
            return "Low confidence"


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = OutcomePredictor()

    test_cases = [
        {
            "text": "The petitioner challenges the detention order as illegal. "
                    "The court finds the detention to be in violation of Article 21. "
                    "The writ petition is allowed and the petitioner is released.",
            "keywords": "habeas corpus fundamental rights detention",
        },
        {
            "text": "The appellant challenges the conviction for murder under IPC 302. "
                    "The evidence on record clearly establishes guilt beyond reasonable doubt. "
                    "The appeal against conviction is dismissed.",
            "keywords": "criminal appeal murder conviction IPC",
        },
    ]

    for i, case in enumerate(test_cases, 1):
        result = predictor.predict(case["text"], case["keywords"])
        print(f"\n🧠 Case {i}:")
        print(f"   Predicted outcome : {result['predicted_outcome'].upper()}")
        print(f"   Confidence        : {result['confidence']:.1%} ({result['confidence_label']})")
        print(f"   All probabilities : {result['all_probabilities']}")