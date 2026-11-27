"""clean_subscribers.py -- cleans data/raw/subscribers.csv and writes data/clean/subscribers.csv.
Run:  python src/clean_subscribers.py
No missing value is ever filled in; cleaning only removes duplicate rows, standardises text values,
and converts impossible ages into missing values.

Rules applied, in order:
  1. Remove exact duplicate rows.
  2. Standardise city names to one of the 7 canonical cities (case/whitespace variants and known aliases).
  3. Standardise auto_renew to exactly "Yes" or "No" (case variants and single-letter codes).
  4. Set age to missing (NaN) wherever it is outside the plausible range of 10-100.
"""
import pandas as pd

from data_cleaner import BaseCleaner

CITY_MAP = {
    "chennai": "Chennai", "madras": "Chennai",
    "mumbai": "Mumbai", "bombay": "Mumbai",
    "bengaluru": "Bengaluru", "bangalore": "Bengaluru",
    "delhi": "Delhi", "new delhi": "Delhi",
    "hyderabad": "Hyderabad", "hyd": "Hyderabad",
    "pune": "Pune",
    "kolkata": "Kolkata", "calcutta": "Kolkata",
}
AUTO_RENEW_MAP = {"yes": "Yes", "y": "Yes", "no": "No", "n": "No"}


class SubscribersCleaner(BaseCleaner):
    def __init__(self):
        super().__init__("subscribers.csv", "subscribers.csv")

    def clean(self, df):
        rules = []

        # Remove exact duplicate rows first
        before = len(df)
        df = df.drop_duplicates()
        rules.append(("Removed exact duplicate rows", before - len(df)))

        # Standardise city names to one of the 7 canonical cities
        std_city = df["city"].astype(str).str.strip().str.lower().map(CITY_MAP)
        changed = (std_city != df["city"]).sum()
        df["city"] = std_city
        rules.append(("Standardised city to one of 7 canonical names", int(changed)))

        # Standardise auto_renew to exactly "Yes" or "No"
        std_auto = df["auto_renew"].astype(str).str.strip().str.lower().map(AUTO_RENEW_MAP)
        changed = (std_auto != df["auto_renew"]).sum()
        df["auto_renew"] = std_auto
        rules.append(("Standardised auto_renew to Yes/No", int(changed)))

        # Set age to missing (NaN) wherever it is outside the plausible range of 10-100
        age = pd.to_numeric(df["age"], errors="coerce")
        out_of_range = age.notna() & ((age < 10) | (age > 100))
        df["age"] = age.mask(out_of_range)
        rules.append(("Set age to missing where outside 10-100", int(out_of_range.sum())))

        return df, rules

    def summary_lines(self, df):
        return [
            f"Cities: {sorted(df['city'].dropna().unique())}",
            f"auto_renew values: {sorted(df['auto_renew'].dropna().unique())}",
        ]


if __name__ == "__main__":
    SubscribersCleaner().run()
