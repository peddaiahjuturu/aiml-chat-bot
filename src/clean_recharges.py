"""clean_recharges.py -- cleans data/raw/recharges.csv and writes data/clean/recharges.csv.
Run:  python src/clean_subscribers.py   (first, to produce data/clean/subscribers.csv)
      python src/clean_recharges.py
No missing value is ever filled in; cleaning only removes rows and standardises existing values.

Rules applied, in order:
  1. Remove exact duplicate rows.
  2. Remove recharges whose subscriber_id does not exist in the cleaned subscribers table.
  3. Lower-case the status column.
  4. Rebuild amount_inr as a number, stripping a leading "INR " text prefix where present.
  5. Parse recharge_date into an actual date type.
"""
import os
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "recharges.csv")
SUBSCRIBERS_CLEAN_PATH = os.path.join("data", "clean", "subscribers.csv")
SUBSCRIBERS_RAW_PATH = os.path.join("data", "raw", "subscribers.csv")
CLEAN_DIR = os.path.join("data", "clean")
CLEAN_PATH = os.path.join(CLEAN_DIR, "recharges.csv")


def clean(path, subscribers_path):
    df = pd.read_csv(path)
    valid_ids = set(pd.read_csv(subscribers_path)["subscriber_id"])
    rules = []

    before = len(df)
    df = df.drop_duplicates()
    rules.append(("Removed exact duplicate rows", before - len(df)))

    before = len(df)
    df = df[df["subscriber_id"].isin(valid_ids)]
    rules.append(("Removed recharges for unknown subscribers", before - len(df)))

    status_lower = df["status"].astype(str).str.lower()
    changed = (status_lower != df["status"]).sum()
    df["status"] = status_lower
    rules.append(("Lower-cased status", int(changed)))

    amount_text = df["amount_inr"].astype(str).str.strip()
    rebuilt = amount_text.str.contains(r"[A-Za-z]", regex=True)
    df["amount_inr"] = pd.to_numeric(amount_text.str.replace(r"[A-Za-z]+", "", regex=True).str.strip(), errors="coerce")
    rules.append(("Rebuilt amount_inr as a number (stripped 'INR ' text)", int(rebuilt.sum())))

    parsed_date = pd.to_datetime(df["recharge_date"], errors="coerce")
    failed = parsed_date.isna() & df["recharge_date"].notna()
    df["recharge_date"] = parsed_date
    rules.append(("Parsed recharge_date to a date type (unparsable values set aside)", int(failed.sum())))

    return df, rules


if __name__ == "__main__":
    subs_path = SUBSCRIBERS_CLEAN_PATH if os.path.exists(SUBSCRIBERS_CLEAN_PATH) else SUBSCRIBERS_RAW_PATH
    clean_df, rules = clean(RAW_PATH, subs_path)
    os.makedirs(CLEAN_DIR, exist_ok=True)
    clean_df.to_csv(CLEAN_PATH, index=False)

    print(f"Rows: {len(clean_df)}")
    print(f"status values: {sorted(clean_df['status'].dropna().unique())}\n")
    for name, count in rules:
        print(f"- {name}: {count}")
