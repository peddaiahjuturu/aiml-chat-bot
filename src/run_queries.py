"""run_queries.py -- runs the four queries in sql/queries.sql against data/tataplay.db and prints results.
Run:  python src/run_queries.py
Requires data/tataplay.db to already exist (run store_db.py first).
"""
import re
import sqlite3

import pandas as pd

from constants import DB_PATH, SQL_QUERIES_PATH as QUERIES_PATH


def load_queries():
    with open(QUERIES_PATH) as f:
        text = f.read()
    # Split the SQL file into chunks, each starting with a query comment like "-- Q1:"
    chunks = re.split(r"(?=-- Q\d+:)", text)
    queries = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # Extract the label from the first line of the chunk, which is the query comment
        label = chunk.splitlines()[0].lstrip("- ").rstrip(".")
        queries.append((label, chunk))
    return queries


def run():
    conn = sqlite3.connect(DB_PATH)
    try:
        queries = load_queries()

        label, sql = queries[0]
        print(f"{label}\n")
        q1 = pd.read_sql_query(sql, conn)
        print(q1.to_string(index=False))
        subscribers = pd.read_sql_query("SELECT * FROM subscribers", conn)
        plans = pd.read_sql_query("SELECT * FROM plans", conn)
        recharges = pd.read_sql_query("SELECT * FROM recharges WHERE status = 'success'", conn)
        merged = recharges.merge(subscribers, on="subscriber_id").merge(plans, on="plan_id")
        check_plan = q1.iloc[0]["plan_name"]
        gb = merged.groupby("plan_name")["amount_inr"].agg(["sum", "mean"]).loc[check_plan]
        print(f"\nPandas groupby confirmation for '{check_plan}': sum={gb['sum']:.2f}, avg={gb['mean']:.2f}")

        label, sql = queries[1]
        print(f"\n{label}\n")
        q2 = pd.read_sql_query(sql, conn)
        zero_tenure = int((q2["tenure_months"] == 0).sum())
        print(f"{len(q2)} subscribers, of whom {zero_tenure} have tenure_months = 0")
        print("An INNER JOIN would silently hide these rows because it only keeps subscribers that have at "
              "least one matching successful recharge; subscribers with zero successful recharges have no "
              "matching row in recharges, so an INNER JOIN drops them from the result entirely instead of "
              "showing them with a NULL recharge.")

        label, sql = queries[2]
        print(f"\n{label}\n")
        q3 = pd.read_sql_query(sql, conn)
        rank1_counts = q3.loc[q3["rank_"] == 1].groupby("city")["subscriber_id"].count()
        print(f"Subscribers with RANK() = 1 per city:\n{rank1_counts.to_string()}")
        print(f"\nKolkata's first four rows:\n{q3.loc[q3['city'] == 'Kolkata'].head(4).to_string(index=False)}")
        print("\nUse RANK() for 'the top spender per city': DENSE_RANK() would list every subscriber tied for "
              "the top amount as rank 1 but never skip a rank for the ties below, and ROW_NUMBER() would "
              "arbitrarily split ties and only ever return a single row per city, so both can mis-identify "
              "or under/over-count the actual top spender(s).")

        label, sql = queries[3]
        print(f"\n{label}\n")
        q4 = pd.read_sql_query(sql, conn)
        print(q4.to_string(index=False))
        print(f"\nRows: {len(q4)}, running total ends at {q4['running_total'].iloc[-1]:.2f}")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
