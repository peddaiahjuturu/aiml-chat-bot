"""churn_features.py -- builds the Part B churn-prediction modelling table from the Part A clean tables.

Joins each subscriber (data/clean/subscribers.csv) to three recharge features computed from
data/clean/recharges.csv as of the snapshot date 2026-03-31:
  n_success_6m   -- number of successful recharges
  n_failed_6m    -- number of failed recharges
  recency_days   -- days from the snapshot to the subscriber's last successful recharge
                    (missing if the subscriber has no successful recharge)
Subscribers with no recharge rows at all get 0 for n_success_6m and n_failed_6m.
"""
import os

import pandas as pd

SNAPSHOT = pd.Timestamp("2026-03-31")
CLEAN_DIR = os.path.join("data", "clean")
SUBSCRIBERS_PATH = os.path.join(CLEAN_DIR, "subscribers.csv")
RECHARGES_PATH = os.path.join(CLEAN_DIR, "recharges.csv")

TARGET = "churned"
NUMERIC_FEATURES = [
    "tenure_months", "age", "add_on_count", "monthly_recharge_inr",
    "hours_watched_per_week", "hd_hours_per_week", "sports_hours_per_week",
    "channels_watched", "support_calls_90d", "n_success_6m", "n_failed_6m", "recency_days",
]
CATEGORICAL_FEATURES = ["city", "plan_id", "payment_mode", "auto_renew"]
LEAK_NUMERIC_FEATURES = ["next_cycle_recharge_inr"]
LEAK_CATEGORICAL_FEATURES = ["closure_request_logged"]


def build_feature_table(subscribers_path=SUBSCRIBERS_PATH, recharges_path=RECHARGES_PATH):
    subs = pd.read_csv(subscribers_path)
    rech = pd.read_csv(recharges_path)
    rech["recharge_date"] = pd.to_datetime(rech["recharge_date"])
    rech = rech[rech["recharge_date"] <= SNAPSHOT]

    success = rech[rech["status"] == "success"]
    failed = rech[rech["status"] == "failed"]

    n_success = success.groupby("subscriber_id").size().rename("n_success_6m")
    n_failed = failed.groupby("subscriber_id").size().rename("n_failed_6m")
    last_success = success.groupby("subscriber_id")["recharge_date"].max().rename("last_success_date")

    df = subs.merge(n_success, on="subscriber_id", how="left")
    df = df.merge(n_failed, on="subscriber_id", how="left")
    df = df.merge(last_success, on="subscriber_id", how="left")

    df["n_success_6m"] = df["n_success_6m"].fillna(0).astype(int)
    df["n_failed_6m"] = df["n_failed_6m"].fillna(0).astype(int)
    df["recency_days"] = (SNAPSHOT - df["last_success_date"]).dt.days
    df = df.drop(columns=["last_success_date"])

    return df
