"""churn_features.py -- builds the Part B churn-prediction modelling table from the Part A clean tables.

Joins each subscriber (data/clean/subscribers.csv) to three recharge features computed from
data/clean/recharges.csv as of the snapshot date 2026-03-31:
  n_success_6m   -- number of successful recharges
  n_failed_6m    -- number of failed recharges
  recency_days   -- days from the snapshot to the subscriber's last successful recharge
                    (missing if the subscriber has no successful recharge)
Subscribers with no recharge rows at all get 0 for n_success_6m and n_failed_6m.
"""
import pandas as pd

from constants import RECHARGES_CLEAN_PATH as RECHARGES_PATH, SNAPSHOT, SUBSCRIBERS_CLEAN_PATH as SUBSCRIBERS_PATH

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
    # Convert recharge_date to datetime and filter for recharges up to the snapshot date
    rech["recharge_date"] = pd.to_datetime(rech["recharge_date"])
    # Keep only recharges up to the snapshot date
    rech = rech[rech["recharge_date"] <= SNAPSHOT]

    # Keep only successful recharges up to the snapshot date
    success = rech[rech["status"] == "success"]
    # Keep only failed recharges up to the snapshot date
    failed = rech[rech["status"] == "failed"]

    # Compute the three recharge features for each subscriber
    # Number of successful recharges
    n_success = success.groupby("subscriber_id").size().rename("n_success_6m")
    # Number of failed recharges
    n_failed = failed.groupby("subscriber_id").size().rename("n_failed_6m")
    # Date of last successful recharge (to compute recency_days)
    last_success = success.groupby("subscriber_id")["recharge_date"].max().rename("last_success_date")

    
    # Merge the number of successful recharges
    df = subs.merge(n_success, on="subscriber_id", how="left")
    # Merge the number of failed recharges
    df = df.merge(n_failed, on="subscriber_id", how="left")
    # Merge the date of last successful recharge
    df = df.merge(last_success, on="subscriber_id", how="left")

    # Fill missing values for the number of successful recharges
    # Fill missing values for the number of failed recharges
    # Calculate recency_days based on the last successful recharge date
    df["n_success_6m"] = df["n_success_6m"].fillna(0).astype(int)
    df["n_failed_6m"] = df["n_failed_6m"].fillna(0).astype(int)
    df["recency_days"] = (SNAPSHOT - df["last_success_date"]).dt.days
    # Drop the last_success_date column as it is no longer needed
    df = df.drop(columns=["last_success_date"])

    return df
