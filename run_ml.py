# run_ml.py
"""
Runs all ML feature steps in order.
Usage: python run_ml.py
"""

import subprocess, sys, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

steps = [
    ("Summarize sample cases",    "src/ml/summarizer.py"),
    ("Train outcome classifier",  "src/ml/train_classifier.py"),
    ("Build citation graph",      "src/ml/citation_graph.py"),
    ("Analyze judge patterns",    "src/ml/judge_analysis.py"),
]

for name, script in steps:
    logger.info(f"\n{'='*50}\n  {name}\n{'='*50}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        logger.error(f"❌ Failed: {script}")
        sys.exit(1)
    logger.info(f"✅ Done: {name}")

logger.info("\n🎉 All ML features built!")