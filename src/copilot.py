"""copilot.py -- Part D task E: the Retention Copilot. Builds and validates a retention brief for each
of the 15 subscribers in outputs/retention_priority.csv, using Part A's database (data/tataplay.db),
Part B's saved model (outputs/churn_model.joblib), and Part C's segments/priority list.
Run:  python src/copilot.py
Writes outputs/retention_briefs.jsonl.
"""
import json
import os
import sqlite3
from typing import Literal

import joblib
import pandas as pd
from pydantic import BaseModel, ValidationError

import constants  # noqa: F401
from churn_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_subscriber_row_from_db
from constants import DB_PATH, OUTPUT_DIR, MODEL_PATH, PRIORITY_PATH, RETENTION_BRIEFS_PATH, SEGMENTS_PATH
from llm_client import call_llm
from policy_rag import build_collection, load_policy_texts, retrieve

MAX_RETRIES = 2
POLICY_FILES = {
    "cancellation_policy.txt", "plan_change_policy.txt", "recharge_and_refund_policy.txt",
    "retention_offers_policy.txt", "technical_support_policy.txt",
}
RETENTION_SYSTEM = (
    "RETENTION BRIEF. Reply with one JSON object only, using only the policy context below -- do not "
    "invent any policy that is not in the context."
)


class RetentionBrief(BaseModel):
    subscriber_id: str
    risk_band: Literal["High", "Medium", "Low"]
    recommended_action: str
    policy_source: str


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def risk_band_for(probability):
    """High if >= 0.40, Medium if >= 0.20, otherwise Low."""
    if probability >= 0.40:
        return "High"
    if probability >= 0.20:
        return "Medium"
    return "Low"


def build_user_message(subscriber_id, risk_band, tenure_months, n_failed, segment, chunks):
    facts_lines = "\n".join([
        f"subscriber_id: {subscriber_id}",
        f"risk_band: {risk_band}",
        f"tenure_months: {int(tenure_months)}",
        f"failed_recharges_6m: {int(n_failed)}",
        f"segment: {segment}",
    ])
    context_block = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    return f"FACTS:\n{facts_lines}\n\nPOLICY CONTEXT:\n{context_block}"


def build_brief(subscriber_id, conn, pipeline, segments_df, collection, file_probability):
    """Builds and validates a retention brief for one subscriber: re-scores with the saved pipeline
    (must agree with the saved probability to 4 decimals), retrieves policy context, calls the model,
    and validates/retries with Pydantic + consistency checks. Returns a result dict; never raises."""
    row = build_subscriber_row_from_db(conn, subscriber_id)
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    rescored_proba = float(pipeline.predict_proba(row[cols])[:, 1][0])
    assert round(rescored_proba, 4) == round(float(file_probability), 4), (
        f"re-scored probability {rescored_proba:.4f} disagrees with the saved {file_probability:.4f} "
        f"for {subscriber_id}"
    )

    risk_band = risk_band_for(file_probability)
    tenure_months = int(row.loc[0, "tenure_months"])
    n_failed = int(row.loc[0, "n_failed_6m"])
    segment = segments_df.loc[segments_df["subscriber_id"] == subscriber_id, "segment_name"].iloc[0]

    query = (f"retention offer for a {risk_band} risk subscriber with {tenure_months} months tenure "
             f"and {n_failed} failed recharges")
    chunks = retrieve(collection, query, k=3)
    user_message = build_user_message(subscriber_id, risk_band, tenure_months, n_failed, segment, chunks)
    messages = [{"role": "system", "content": RETENTION_SYSTEM}, {"role": "user", "content": user_message}]

    facts = {
        "subscriber_id": subscriber_id, "risk_band": risk_band, "tenure_months": tenure_months,
        "failed_recharges_6m": n_failed, "segment": segment, "churn_probability": round(file_probability, 6),
    }

    for attempt in range(1, MAX_RETRIES + 2):
        reply = call_llm(messages)
        try:
            brief = RetentionBrief(**json.loads(reply))
            if brief.subscriber_id != subscriber_id:
                raise ValueError(f"subscriber_id mismatch: got {brief.subscriber_id!r}, expected {subscriber_id!r}")
            if brief.risk_band != risk_band:
                raise ValueError(f"risk_band mismatch: got {brief.risk_band!r}, expected {risk_band!r}")
            if brief.policy_source not in POLICY_FILES:
                raise ValueError(f"policy_source {brief.policy_source!r} is not a real policy file")
            return {"facts": facts, "status": "valid", "attempts": attempt, "brief": brief.model_dump()}
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            if attempt == MAX_RETRIES + 1:
                return {"facts": facts, "status": "manual_review", "attempts": attempt, "brief": None}
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"VALIDATION ERROR: {exc}"})


def main():
    # Load necessary data and models for building retention briefs
    priority = pd.read_csv(PRIORITY_PATH)
    segments = pd.read_csv(SEGMENTS_PATH)
    pipeline = joblib.load(MODEL_PATH)
    collection = build_collection(load_policy_texts(), "copilot_policies", size=400, overlap=60)

    conn = sqlite3.connect(DB_PATH)
    results = []
    for _, priority_row in priority.iterrows():
        result = build_brief(
            priority_row["subscriber_id"], conn, pipeline, segments, collection, priority_row["churn_probability"]
        )
        results.append(result)
    conn.close()

    section("Task E: retention briefs summary")
    valid = [r for r in results if r["status"] == "valid"]
    print(f"Valid briefs: {len(valid)} of {len(results)}")
    print(f"Per risk_band:\n{pd.Series([r['facts']['risk_band'] for r in results]).value_counts().to_string()}")
    actions = [r["brief"]["recommended_action"] if r["brief"] else "manual_review" for r in results]
    print(f"Per recommended_action:\n{pd.Series(actions).value_counts().to_string()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RETENTION_BRIEFS_PATH, "w") as f:
        for r in results:
            f.write(json.dumps({
                "subscriber_id": r["facts"]["subscriber_id"], "status": r["status"], "attempts": r["attempts"],
                "facts": r["facts"], "brief": r["brief"],
            }) + "\n")
    print(f"\nWrote {RETENTION_BRIEFS_PATH} ({len(results)} rows)")


if __name__ == "__main__":
    main()
