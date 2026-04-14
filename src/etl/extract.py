# src/etl/extract.py
"""
EXTRACT STEP:
Reads all PDF files from data/raw/supreme_court_pdfs/
Extracts: case title, year, full text, file path
Saves raw extracted data to data/processed/cases_raw.csv
"""

import fitz  # PyMuPDF
import os
import pandas as pd
from tqdm import tqdm
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a single PDF file."""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        return full_text.strip()
    except Exception as e:
        logger.warning(f"Failed to read {pdf_path}: {e}")
        return ""


def extract_case_title_from_filename(filename: str) -> str:
    """
    Convert filename like 'A_K_Gopalan_vs_T_...' to
    'A K Gopalan vs T ...'
    """
    name = filename.replace(".PDF", "").replace(".pdf", "")
    name = name.replace("_", " ")
    return name.strip()


def extract_year_from_path(pdf_path: str) -> str:
    """Extract year from folder structure: .../1950/case.PDF → '1950'"""
    parts = pdf_path.replace("\\", "/").split("/")
    for part in parts:
        if part.isdigit() and len(part) == 4:
            return part
    return "Unknown"


def extract_court_from_path(pdf_path: str) -> str:
    """Extract court name from folder structure."""
    parts = pdf_path.replace("\\", "/").split("/")
    for part in parts:
        if "court" in part.lower() or "supreme" in part.lower():
            return part.replace("_", " ").title()
    return "Supreme Court of India"


def extract_vs_parties(case_title: str):
    """Try to split 'Petitioner vs Respondent' from case title."""
    patterns = [" vs ", " Vs ", " VS ", " versus ", " v. ", " v "]
    for pat in patterns:
        if pat in case_title:
            parts = case_title.split(pat, 1)
            return parts[0].strip(), parts[1].strip()
    return case_title.strip(), "Unknown"


def extract_all_pdfs(raw_dir: str) -> pd.DataFrame:
    """
    Walk through all subdirectories of raw_dir,
    find all PDF files, extract their text and metadata.
    """
    records = []

    # Find all PDF files recursively
    pdf_files = []
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.upper().endswith(".PDF") or file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    logger.info(f"Found {len(pdf_files)} PDF files in {raw_dir}")

    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
        filename = os.path.basename(pdf_path)
        case_title = extract_case_title_from_filename(filename)
        year = extract_year_from_path(pdf_path)
        court = extract_court_from_path(pdf_path)
        petitioner, respondent = extract_vs_parties(case_title)
        text = extract_text_from_pdf(pdf_path)

        records.append({
            "case_id":        f"{year}_{filename.replace('.PDF','').replace('.pdf','')}",
            "case_title":     case_title,
            "year":           year,
            "court":          court,
            "petitioner":     petitioner,
            "respondent":     respondent,
            "judgment_text":  text,
            "source_file":    pdf_path,
            "text_length":    len(text),
        })

    df = pd.DataFrame(records)
    logger.info(f"Extracted {len(df)} cases. Avg text length: {df['text_length'].mean():.0f} chars")
    return df


if __name__ == "__main__":
    RAW_DIR = "data/raw/supreme_court_pdfs"
    OUT_PATH = "data/processed/cases_raw.csv"

    os.makedirs("data/processed", exist_ok=True)

    df = extract_all_pdfs(RAW_DIR)

    # Save
    df.to_csv(OUT_PATH, index=False)
    logger.info(f"✅ Saved {len(df)} records to {OUT_PATH}")

    # Quick summary
    print("\n📊 Extraction Summary:")
    print(f"   Total cases : {len(df)}")
    print(f"   Years found : {sorted(df['year'].unique())[:5]} ...")
    print(f"   Empty texts : {(df['text_length'] == 0).sum()}")
    print(f"   Columns     : {df.columns.tolist()}")