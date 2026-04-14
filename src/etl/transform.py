# src/etl/transform.py
"""
TRANSFORM STEP:
Reads cases_raw.csv
Cleans text, extracts legal keywords, detects outcome,
extracts citations, prepares final structured dataset.
Saves to data/processed/cases_clean.csv
"""

import pandas as pd
import re
import os
import spacy
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import logging

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load spacy model
nlp = spacy.load("en_core_web_sm", disable=["ner"])  # disable NER for speed

STOP_WORDS = set(stopwords.words("english"))

# Legal keywords to look for
LEGAL_KEYWORDS = [
    "constitution", "article", "section", "act", "judgment", "appeal",
    "petition", "writ", "habeas corpus", "fundamental rights", "supreme court",
    "high court", "tribunal", "bench", "justice", "advocate", "plaintiff",
    "defendant", "respondent", "appellant", "order", "decree", "injunction",
    "acquittal", "conviction", "bail", "custody", "evidence", "witness",
    "contract", "tort", "negligence", "liability", "damages", "property",
    "civil", "criminal", "constitutional", "statutory", "jurisdiction"
]

# Outcome detection patterns
OUTCOME_PATTERNS = {
    "allowed":    [r"\bappeal\s+(?:is\s+)?allowed\b", r"\bpetition\s+(?:is\s+)?allowed\b",
                   r"\ballow(?:ed)?\b", r"\bin\s+favour\s+of\s+(?:the\s+)?appellant"],
    "dismissed":  [r"\bappeal\s+(?:is\s+)?dismissed\b", r"\bpetition\s+(?:is\s+)?dismissed\b",
                   r"\bdismiss(?:ed)?\b", r"\bin\s+favour\s+of\s+(?:the\s+)?respondent"],
    "partly allowed": [r"\bpartly\s+allowed\b", r"\bpartially\s+allowed\b"],
    "remanded":   [r"\bremand(?:ed)?\b", r"\bsent\s+back\b"],
}


def clean_text(text: str) -> str:
    """Remove noise from extracted PDF text."""
    if not isinstance(text, str):
        return ""
    # Remove excessive whitespace and newlines
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers like "Page 1 of 10"
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    # Remove headers/footers (short repeated lines)
    text = re.sub(r'\d{1,3}\s*$', '', text)
    # Remove special characters but keep legal punctuation
    text = re.sub(r'[^\w\s.,;:()\-\'/]', ' ', text)
    return text.strip()


def extract_first_n_words(text: str, n: int = 200) -> str:
    """Extract facts section — typically in first part of judgment."""
    words = text.split()
    return " ".join(words[:n])


def extract_legal_keywords(text: str) -> str:
    """Find which legal keywords appear in the case text."""
    text_lower = text.lower()
    found = [kw for kw in LEGAL_KEYWORDS if kw in text_lower]
    return ", ".join(found)


def detect_outcome(text: str) -> str:
    """Detect case outcome from judgment text using regex patterns."""
    text_lower = text.lower()
    # Check last 1000 chars first — outcome is usually at the end
    tail = text_lower[-1000:]

    for outcome, patterns in OUTCOME_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, tail, re.IGNORECASE):
                return outcome

    # Try full text if not found in tail
    for outcome, patterns in OUTCOME_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower, re.IGNORECASE):
                return outcome

    return "unknown"


def extract_citations(text: str) -> str:
    """
    Extract case citations like:
    - AIR 1950 SC 27
    - (2001) 3 SCC 756
    - 1984 SCR (2) 67
    """
    patterns = [
        r'AIR\s+\d{4}\s+SC\s+\d+',
        r'\(\d{4}\)\s+\d+\s+SCC\s+\d+',
        r'\d{4}\s+SCR\s+\(\d+\)\s+\d+',
        r'\d{4}\s+\(\d+\)\s+SCC\s+\d+',
    ]
    citations = []
    for pat in patterns:
        found = re.findall(pat, text)
        citations.extend(found)
    return " | ".join(list(set(citations)))  # deduplicate


def extract_judges(text: str) -> str:
    """Extract judge names — typically in first 500 chars of judgment."""
    head = text[:500]
    # Pattern: JUSTICE/J. followed by name
    judges = re.findall(
        r'(?:JUSTICE|Justice|HON\'BLE|J\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        head
    )
    return ", ".join(list(set(judges)))


def transform_dataset(raw_path: str, clean_path: str):
    """Full transformation pipeline."""
    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} cases")

    # Drop rows with empty text
    df = df[df["text_length"] > 100].copy()
    logger.info(f"After dropping empty: {len(df)} cases")

    tqdm.pandas(desc="Cleaning text")
    df["judgment_text_clean"] = df["judgment_text"].progress_apply(clean_text)

    tqdm.pandas(desc="Extracting case facts")
    df["case_facts"] = df["judgment_text_clean"].progress_apply(
        lambda t: extract_first_n_words(t, 200)
    )

    tqdm.pandas(desc="Detecting outcomes")
    df["outcome"] = df["judgment_text_clean"].progress_apply(detect_outcome)

    tqdm.pandas(desc="Extracting citations")
    df["citations"] = df["judgment_text"].progress_apply(extract_citations)

    tqdm.pandas(desc="Extracting legal keywords")
    df["legal_keywords"] = df["judgment_text_clean"].progress_apply(extract_legal_keywords)

    tqdm.pandas(desc="Extracting judges")
    df["judges"] = df["judgment_text"].progress_apply(extract_judges)

    # Create a short summary field (first 512 chars of clean text)
    df["text_preview"] = df["judgment_text_clean"].str[:512]

    # Final column selection
    final_cols = [
        "case_id", "case_title", "year", "court",
        "petitioner", "respondent", "judges",
        "case_facts", "judgment_text_clean", "text_preview",
        "outcome", "citations", "legal_keywords",
        "source_file", "text_length"
    ]
    df = df[final_cols]

    # Save
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df.to_csv(clean_path, index=False)

    # Summary
    logger.info(f"✅ Saved clean data to {clean_path}")
    print("\n📊 Transform Summary:")
    print(f"   Total cases      : {len(df)}")
    print(f"   Outcome counts   :\n{df['outcome'].value_counts()}")
    print(f"   Cases with citations: {(df['citations'] != '').sum()}")
    print(f"   Columns          : {df.columns.tolist()}")

    return df


if __name__ == "__main__":
    transform_dataset(
        raw_path="data/processed/cases_raw.csv",
        clean_path="data/processed/cases_clean.csv"
    )