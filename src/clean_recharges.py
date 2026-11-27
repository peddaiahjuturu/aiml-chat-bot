"""clean_recharges.py -- cleans data/raw/recharges.csv and writes data/clean/recharges.csv.
Run:  python src/clean_subscribers.py   (first, to produce data/clean/subscribers.csv)
      python src/clean_recharges.py
No missing value is ever filled in; cleaning only removes rows and standardises existing values.

Rules applied, in order:
  1. Remove exact duplicate rows.
  2. Remove recharges whose subscriber_id does not exist in the cleaned subscribers table.
  3. Lower-case the status column.
  4. Convert amount_inr to a number; where it cannot be converted (e.g. "INR 498"), rebuild it
     as that subscriber's monthly_recharge_inr from the subscribers table.
  5. Parse recharge_date into an actual date type.
"""
import os
import pandas as pd

from constants import SUBSCRIBERS_CLEAN_PATH, SUBSCRIBERS_RAW_PATH
from data_cleaner import BaseCleaner


class RechargesCleaner(BaseCleaner):
    def __init__(self):
        super().__init__("recharges.csv", "recharges.csv")

    def clean(self, df):
        rules = []
        subscribers_path = SUBSCRIBERS_CLEAN_PATH if os.path.exists(SUBSCRIBERS_CLEAN_PATH) else SUBSCRIBERS_RAW_PATH
        subscribers_df = pd.read_csv(subscribers_path)
        valid_ids = set(subscribers_df["subscriber_id"])
        monthly_recharge = subscribers_df.set_index("subscriber_id")["monthly_recharge_inr"]

        # Remove exact duplicate rows first
        before = len(df)
        df = df.drop_duplicates()
        rules.append(("Removed exact duplicate rows", before - len(df)))

        # Remove recharges whose subscriber_id does not exist in the cleaned subscribers table
        before = len(df)
        df = df[df["subscriber_id"].isin(valid_ids)]
        rules.append(("Removed recharges for unknown subscribers", before - len(df)))

        # Lower-case the status column
        status_lower = df["status"].astype(str).str.lower()
        changed = (status_lower != df["status"]).sum()
        df["status"] = status_lower
        rules.append(("Lower-cased status", int(changed)))

        # Convert amount_inr to a number; where it cannot be converted, rebuild it as the
        # subscriber's monthly_recharge_inr
        numeric_amount = pd.to_numeric(df["amount_inr"], errors="coerce")
        unparsable = numeric_amount.isna()
        df["amount_inr"] = numeric_amount.where(~unparsable, df["subscriber_id"].map(monthly_recharge))
        rules.append(("Rebuilt amount_inr from the subscriber's monthly_recharge_inr", int(unparsable.sum())))

        # Parse recharge_date into an actual date type
        parsed_date = pd.to_datetime(df["recharge_date"], errors="coerce")
        failed = parsed_date.isna() & df["recharge_date"].notna()
        df["recharge_date"] = parsed_date
        rules.append(("Parsed recharge_date to a date type (unparsable values set aside)", int(failed.sum())))

        return df, rules

    def summary_lines(self, df):
        status_counts = df["status"].value_counts().to_dict()
        total_success = df.loc[df["status"] == "success", "amount_inr"].sum()
        return [
            f"status values: {sorted(df['status'].dropna().unique())}",
            f"status counts: {status_counts}",
            f"Total successful recharge amount (INR): {total_success:.0f}",
        ]


if __name__ == "__main__":
    RechargesCleaner().run()
