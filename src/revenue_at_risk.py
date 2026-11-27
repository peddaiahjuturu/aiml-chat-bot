"""revenue_at_risk.py -- Part C task 4: revenue-at-risk priority list for Part B's test subscribers.
Run:  python src/revenue_at_risk.py
Reads data/clean/subscribers.csv (Part A), outputs/test_scores.csv (Part B) and outputs/segments.csv (Part C).
This priority list is built from test-set subscribers whose outcome is already known, so it illustrates the
method; it is not a forward-looking evaluation. Writes outputs/retention_priority.csv.
"""
import os

import pandas as pd

from constants import OUTPUT_DIR, PRIORITY_PATH, SCORES_PATH, SEGMENTS_PATH, SUBSCRIBERS_CLEAN_PATH
from spend_model import SPEND_FEATURES, build_best_ridge_pipeline

TOP_N = 15


def main():
    subs = pd.read_csv(SUBSCRIBERS_CLEAN_PATH)
    scores = pd.read_csv(SCORES_PATH)
    segments = pd.read_csv(SEGMENTS_PATH)

    ridge_pipe = build_best_ridge_pipeline(subs)

    subs_features = subs[["subscriber_id"] + SPEND_FEATURES]
    merged = scores.merge(subs_features, on="subscriber_id", how="left").merge(segments, on="subscriber_id", how="left")
    merged["predicted_next_recharge_inr"] = ridge_pipe.predict(merged[SPEND_FEATURES])
    merged["revenue_at_risk_inr"] = merged["churn_probability"] * merged["predicted_next_recharge_inr"]

    priority = merged.sort_values("revenue_at_risk_inr", ascending=False).head(TOP_N)
    priority = priority[[
        "subscriber_id", "churn_probability", "predicted_next_recharge_inr", "revenue_at_risk_inr",
        "segment_name", "churned",
    ]]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    priority.to_csv(PRIORITY_PATH, index=False)

    n_churned = int(priority["churned"].sum())
    print(f"Wrote {PRIORITY_PATH} ({len(priority)} rows)")
    print(f"Sum of revenue at risk (top {TOP_N}): INR {priority['revenue_at_risk_inr'].sum():,.0f}")
    print(f"{n_churned} of {len(priority)} actually churned (bar: at least 9)")
    assert n_churned >= 9, "priority list fails the 'at least 9 churned' bar"


if __name__ == "__main__":
    main()
