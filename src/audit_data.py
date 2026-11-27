"""audit_data.py -- prints shape, dtypes, and missing-value counts/percentages for the raw subscribers and recharges tables.
Run:  python src/audit_data.py"""
import os
import pandas as pd

SUBSCRIBERS_PATH = os.path.join("data", "raw", "subscribers.csv")
RECHARGES_PATH = os.path.join("data", "raw", "recharges.csv")


def audit(path):
    df = pd.read_csv(path)
    print(f"Table: {path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing_count": missing, "missing_pct": pct})
    print(report.to_string())
    return report


if __name__ == "__main__":
    audit(SUBSCRIBERS_PATH)
    print()
    audit(RECHARGES_PATH)
