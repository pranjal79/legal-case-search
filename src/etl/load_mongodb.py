# src/etl/load_mongodb.py
"""
LOAD STEP:
Reads cleaned CSV and loads all cases into MongoDB.
MongoDB acts as our main database for the app.
"""

import pandas as pd
from pymongo import MongoClient, ASCENDING
from pymongo.errors import BulkWriteError
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_db():
    """Connect to MongoDB and return the database."""
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    client = MongoClient(uri)
    db = client["legal_cases_db"]
    return db


def load_to_mongodb(clean_path: str):
    """Load cleaned cases CSV into MongoDB."""
    logger.info(f"Reading {clean_path}")
    df = pd.read_csv(clean_path)

    # Convert NaN to None (MongoDB-friendly)
    df = df.where(pd.notna(df), None)

    # Convert to list of dicts
    records = df.to_dict(orient="records")

    db = get_db()
    collection = db["cases"]

    # Drop existing collection to reload fresh
    collection.drop()
    logger.info("Dropped existing collection")

    # Insert in batches of 500
    batch_size = 500
    total_inserted = 0

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        try:
            result = collection.insert_many(batch, ordered=False)
            total_inserted += len(result.inserted_ids)
        except BulkWriteError as e:
            logger.warning(f"Batch write error: {e.details['nInserted']} inserted in this batch")

    # Create indexes for fast querying
    collection.create_index([("case_id", ASCENDING)], unique=True)
    collection.create_index([("year", ASCENDING)])
    collection.create_index([("outcome", ASCENDING)])
    collection.create_index([("court", ASCENDING)])

    logger.info(f"✅ Inserted {total_inserted} documents into MongoDB")
    logger.info(f"   Indexes created on: case_id, year, outcome, court")

    # Quick verification
    count = collection.count_documents({})
    sample = collection.find_one()
    print(f"\n✅ MongoDB Verification:")
    print(f"   Total documents : {count}")
    print(f"   Sample keys     : {list(sample.keys()) if sample else 'None'}")


if __name__ == "__main__":
    load_to_mongodb("data/processed/cases_clean.csv")