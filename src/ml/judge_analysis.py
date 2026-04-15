# src/ml/judge_analysis.py

"""
JUDGE PATTERN ANALYSIS:
Improved version with judge name cleaning.
"""

import pandas as pd
import json
import os
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# 🧹 CLEAN JUDGE NAME (IMPORTANT FIX)
# ────────────────────────────────────────────────────────────────
def clean_judge_name(judge: str):
    if not isinstance(judge, str):
        return None

    judge = judge.strip()

    # Remove common noise words
    bad_patterns = [
        r"\bbench\b",
        r"\bstate\b",
        r"\bcourt\b",
        r"\bjudges?\b",
        r"\band\b",
        r"\bof\b",
        r"\bthe\b",
        r"\breportable\b",
    ]

    for pattern in bad_patterns:
        judge = re.sub(pattern, "", judge, flags=re.IGNORECASE)

    # Remove extra spaces
    judge = re.sub(r"\s+", " ", judge).strip()

    # Remove very short / invalid names
    if len(judge) < 4:
        return None

    # Remove non-name entries
    if any(char.isdigit() for char in judge):
        return None

    return judge


# ────────────────────────────────────────────────────────────────
# 🧠 MAIN ANALYSIS
# ────────────────────────────────────────────────────────────────
def analyze_judges(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["judges"] = df["judges"].fillna("")
    df["outcome"] = df["outcome"].fillna("unknown")

    # Split multiple judges
    df["judge_list"] = df["judges"].apply(
        lambda j: [x.strip() for x in j.split(",") if x.strip()]
    )

    df_exp = df.explode("judge_list")

    # 🧹 Apply cleaning
    df_exp["judge_list"] = df_exp["judge_list"].apply(clean_judge_name)
    df_exp = df_exp.dropna(subset=["judge_list"])

    if df_exp.empty:
        logger.warning("No valid judge data found — returning empty stats")
        return {}

    stats = {}

    for judge, group in df_exp.groupby("judge_list"):

        outcome_counts = group["outcome"].value_counts().to_dict()
        total = len(group)

        # 🔥 Skip very small samples (optional but recommended)
        if total < 5:
            continue

        # Top keywords
        all_kw = " ".join(group["legal_keywords"].fillna("").tolist())
        kw_freq = {}

        for kw in all_kw.split(","):
            kw = kw.strip().lower()
            if kw:
                kw_freq[kw] = kw_freq.get(kw, 0) + 1

        top_keywords = sorted(kw_freq, key=kw_freq.get, reverse=True)[:5]

        stats[judge] = {
            "total_cases": total,
            "outcome_counts": outcome_counts,
            "allowed_rate": round(outcome_counts.get("allowed", 0) / total, 3),
            "dismissed_rate": round(outcome_counts.get("dismissed", 0) / total, 3),
            "top_keywords": top_keywords,
            "years_active": sorted(group["year"].dropna().unique().tolist()),
        }

    # Sort by most experienced judges
    stats = dict(sorted(stats.items(), key=lambda x: x[1]["total_cases"], reverse=True))

    return stats


# ────────────────────────────────────────────────────────────────
# 💾 SAVE RESULTS
# ────────────────────────────────────────────────────────────────
def save_judge_stats(clean_path: str, out_path: str = "models/judge_stats.json"):
    df = pd.read_csv(clean_path).fillna("")
    stats = analyze_judges(df)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"✅ Judge stats saved to {out_path}")

    print(f"\n👨‍⚖️ Top 5 judges by case count:")
    for judge, s in list(stats.items())[:5]:
        print(
            f"   {judge[:30]:<30} | {s['total_cases']} cases | "
            f"allowed: {s['allowed_rate']:.0%} | dismissed: {s['dismissed_rate']:.0%}"
        )

    return stats


# ────────────────────────────────────────────────────────────────
# 🚀 RUN
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    save_judge_stats("data/processed/cases_clean.csv")