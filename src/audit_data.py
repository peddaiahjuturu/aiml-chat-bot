"""audit_data.py -- prints shape, dtypes, and missing-value counts/percentages for the raw subscribers and recharges tables,
plus the extra data-quality figures (duplicate rows, distinct id/spelling counts, orphan rows) used to size the cleaning rules.
Run:  python src/audit_data.py"""
import pandas as pd

from constants import RECHARGES_RAW_PATH as RECHARGES_PATH, SUBSCRIBERS_RAW_PATH as SUBSCRIBERS_PATH


def audit(path):
    df = pd.read_csv(path)
    print(f"Table: {path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    # Print column data types and missing values
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing_count": missing, "missing_pct": pct})
    print(report.to_string())
    return df, report


def audit_subscribers(df):
    age = pd.to_numeric(df["age"], errors="coerce")
    print(f"Exact duplicate rows: {df.duplicated().sum()}")
    print(f"Distinct subscriber_id: {df['subscriber_id'].nunique()}")
    print(f"Blank age: {df['age'].isna().sum()}")
    print(f"age values outside 10-100: {int((age.notna() & ((age < 10) | (age > 100))).sum())}")
    print(f"Blank payment_mode: {df['payment_mode'].isna().sum()}")
    print(f"Distinct raw city spellings: {df['city'].nunique()}")
    print(f"Distinct raw auto_renew spellings: {df['auto_renew'].nunique()}")
    print(f"Blank next_cycle_recharge_inr: {df['next_cycle_recharge_inr'].isna().sum()}")


def audit_recharges(df, subscribers_df):
    not_plain_number = pd.to_numeric(df["amount_inr"], errors="coerce").isna().sum()
    valid_ids = set(subscribers_df["subscriber_id"])
    print(f"Exact duplicate rows: {df.duplicated().sum()}")
    print(f"Distinct recharge_id: {df['recharge_id'].nunique()}")
    print(f"amount_inr values that are not plain numbers: {int(not_plain_number)}")
    print(f"Distinct raw status spellings: {df['status'].nunique()}")
    print(f"Rows whose subscriber_id does not exist in subscribers.csv: {int((~df['subscriber_id'].isin(valid_ids)).sum())}")


if __name__ == "__main__":
    subs_df, _ = audit(SUBSCRIBERS_PATH)
    print()
    audit_subscribers(subs_df)
    print()
    rech_df, _ = audit(RECHARGES_PATH)
    print()
    audit_recharges(rech_df, subs_df)
