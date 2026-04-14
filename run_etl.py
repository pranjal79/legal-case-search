# run_etl.py
"""
Master ETL runner — runs all 3 steps in order.
Usage: python run_etl.py
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

steps = [
    ("EXTRACT",   "src/etl/extract.py"),
    ("TRANSFORM", "src/etl/transform.py"),
    ("LOAD",      "src/etl/load_mongodb.py"),
]

for name, script in steps:
    logger.info(f"\n{'='*50}")
    logger.info(f"  Running {name} step: {script}")
    logger.info(f"{'='*50}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        logger.error(f"❌ {name} step failed. Stopping pipeline.")
        sys.exit(1)
    logger.info(f"✅ {name} step completed successfully.")

logger.info("\n🎉 Full ETL pipeline completed!")