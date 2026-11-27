"""store_db.py -- writes the three clean tables (subscribers, recharges, plans) to a SQLite database.
Run:  python src/store_db.py
Requires data/clean/subscribers.csv, data/clean/recharges.csv and data/clean/plans.csv to already exist
(run clean_subscribers.py, clean_recharges.py and clean_plans.py first).
"""
import sqlite3

import pandas as pd

from constants import DB_PATH, PLANS_CLEAN_PATH, RECHARGES_CLEAN_PATH, SUBSCRIBERS_CLEAN_PATH

TABLES = {
    "subscribers": SUBSCRIBERS_CLEAN_PATH,
    "recharges": RECHARGES_CLEAN_PATH,
    "plans": PLANS_CLEAN_PATH,
}


def store():
    conn = sqlite3.connect(DB_PATH)
    try:
        for table_name, csv_path in TABLES.items():
            df = pd.read_csv(csv_path)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"Wrote table '{table_name}' ({len(df)} rows) from {csv_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    store()
