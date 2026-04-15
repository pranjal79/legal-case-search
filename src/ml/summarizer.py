# src/ml/summarizer.py
"""
CASE SUMMARIZER:
Uses facebook/bart-large-cnn (or a lighter model for low-RAM machines)
to generate concise summaries of long judgment texts.
"""

import os
import yaml
import logging
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CaseSummarizer:
    """
    Wraps a HuggingFace summarization pipeline.
    Load once, summarize many times.
    """

    # If your machine has < 8GB RAM, use the lighter model below instead:
    # LIGHT_MODEL  = "sshleifer/distilbart-cnn-12-6"   ← faster, smaller
    # HEAVY_MODEL  = "facebook/bart-large-cnn"          ← better quality

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        logger.info(f"Loading summarizer model: {model_name}")
        self.model_name = model_name
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            framework="pt",          # PyTorch
        )
        logger.info("Summarizer ready!")

    def summarize(self, text: str, max_input_chars: int = 2000) -> str:
        """
        Summarize a single case judgment.

        Args:
            text:             Full judgment text
            max_input_chars:  Truncate input to this length (model token limit)

        Returns:
            Summary string
        """
        if not text or len(text.strip()) < 100:
            return "Insufficient text to summarize."

        # Truncate to model's safe input length
        text = text[:max_input_chars]

        try:
            result = self.summarizer(
                text,
                max_length=180,       # summary max tokens
                min_length=60,        # summary min tokens
                do_sample=False,      # greedy decoding = deterministic
                truncation=True,
            )
            return result[0]["summary_text"].strip()
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            # Fallback: return first 3 sentences
            sentences = text.split(". ")
            return ". ".join(sentences[:3]) + "."

    def summarize_batch(self, texts: list, batch_size: int = 8) -> list:
        """Summarize a list of texts efficiently in batches."""
        summaries = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Summarizing"):
            batch = texts[i : i + batch_size]
            # Truncate each text
            batch = [t[:2000] if isinstance(t, str) else "" for t in batch]
            # Filter empty texts
            valid = [(j, t) for j, t in enumerate(batch) if len(t) > 100]
            batch_results = ["Insufficient text."] * len(batch)
            if valid:
                try:
                    idxs, valid_texts = zip(*valid)
                    results = self.summarizer(
                        list(valid_texts),
                        max_length=180,
                        min_length=60,
                        do_sample=False,
                        truncation=True,
                    )
                    for idx, res in zip(idxs, results):
                        batch_results[idx] = res["summary_text"].strip()
                except Exception as e:
                    logger.warning(f"Batch summarization error: {e}")
            summaries.extend(batch_results)
        return summaries


def generate_and_save_summaries(
    clean_path: str,
    out_path: str,
    sample_size: int = 500,          # summarize first N cases to save time
):
    """
    Generate summaries for a sample of cases and save to CSV.
    Full dataset summarization can take hours on CPU.
    """
    df = pd.read_csv(clean_path).fillna("")
    df_sample = df.head(sample_size).copy()

    summarizer = CaseSummarizer()

    texts = df_sample["judgment_text_clean"].tolist()
    summaries = summarizer.summarize_batch(texts)

    df_sample["summary"] = summaries
    df_sample.to_csv(out_path, index=False)
    logger.info(f"✅ Saved {len(df_sample)} summaries to {out_path}")
    return df_sample


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summarizer = CaseSummarizer()

    test_text = """
    The petitioner challenges the constitutional validity of Section 144 of the
    Code of Criminal Procedure on the grounds that it violates Article 19 of
    the Constitution of India which guarantees the right to freedom of speech
    and assembly. The learned counsel argued that the restrictions imposed are
    unreasonable and beyond the scope permitted by the Constitution.
    The respondent state contended that the restrictions are necessary for
    maintaining public order and are saved by the reasonable restrictions
    clause under Article 19(2). After hearing both parties, the court held
    that the restrictions imposed were reasonable and within constitutional
    limits. The petition is accordingly dismissed.
    """

    summary = summarizer.summarize(test_text)
    print(f"\n📝 Test Summary:\n{summary}")