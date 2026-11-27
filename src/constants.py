"""constants.py -- shared paths and parameters used across the data-pipeline and churn-model scripts."""
import os

import pandas as pd

RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")
OUTPUT_DIR = "outputs"
SQL_DIR = "sql"

SUBSCRIBERS_RAW_PATH = os.path.join(RAW_DIR, "subscribers.csv")
RECHARGES_RAW_PATH = os.path.join(RAW_DIR, "recharges.csv")
PLANS_RAW_PATH = os.path.join(RAW_DIR, "plans.csv")

SUBSCRIBERS_CLEAN_PATH = os.path.join(CLEAN_DIR, "subscribers.csv")
RECHARGES_CLEAN_PATH = os.path.join(CLEAN_DIR, "recharges.csv")
PLANS_CLEAN_PATH = os.path.join(CLEAN_DIR, "plans.csv")

DB_PATH = os.path.join("data", "tataplay.db")
SQL_QUERIES_PATH = os.path.join(SQL_DIR, "queries.sql")

MODEL_PATH = os.path.join(OUTPUT_DIR, "churn_model.joblib")
SCORES_PATH = os.path.join(OUTPUT_DIR, "test_scores.csv")

SNAPSHOT = pd.Timestamp("2026-03-31")
RANDOM_STATE = 42
MISS_COST, OFFER_COST = 1200, 300
