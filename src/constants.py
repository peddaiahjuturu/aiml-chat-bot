"""constants.py -- shared paths and parameters used across the data-pipeline and churn-model scripts."""
import os
import sys

import pandas as pd

# Repo root (parent of src/), needed so "import mock_llm" works regardless of the caller's cwd.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")
OUTPUT_DIR = "outputs"
SQL_DIR = "sql"
POLICIES_DIR = os.path.join("data", "policies")

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
SEGMENTS_PATH = os.path.join(OUTPUT_DIR, "segments.csv")
PRIORITY_PATH = os.path.join(OUTPUT_DIR, "retention_priority.csv")
TICKETS_RAW_PATH = os.path.join(RAW_DIR, "tickets.csv")
FEW_SHOT_PATH = os.path.join(RAW_DIR, "few_shot_examples.csv")
POLICY_EVAL_PATH = os.path.join(RAW_DIR, "policy_eval.csv")
TICKET_CLASSIFICATION_PATH = os.path.join(OUTPUT_DIR, "ticket_classification.csv")
TICKET_EXTRACTIONS_PATH = os.path.join(OUTPUT_DIR, "ticket_extractions.jsonl")
RAG_EVAL_PATH = os.path.join(OUTPUT_DIR, "rag_eval.csv")
RETENTION_BRIEFS_PATH = os.path.join(OUTPUT_DIR, "retention_briefs.jsonl")

SNAPSHOT = pd.Timestamp("2026-03-31")
RANDOM_STATE = 42
MISS_COST, OFFER_COST = 1200, 300
