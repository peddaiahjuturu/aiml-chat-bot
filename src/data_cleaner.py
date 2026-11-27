"""data_cleaner.py -- shared base class for the data/raw -> data/clean cleaning scripts.
No missing value is ever filled in by a cleaner; cleaning only removes rows and standardises existing values.
"""
import os
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")


class BaseCleaner:
    """Reads raw_filename from RAW_DIR, applies self.clean(), and writes clean_filename to CLEAN_DIR."""

    def __init__(self, raw_filename, clean_filename):
        self.raw_path = os.path.join(RAW_DIR, raw_filename)
        self.clean_path = os.path.join(CLEAN_DIR, clean_filename)

    def clean(self, df):
        """Return (cleaned_df, rules); rules is a list of (description, affected_row_count). Must be overridden."""
        raise NotImplementedError

    def summary_lines(self, df):
        """Extra lines printed after 'Rows: n' and before the rule list. Override as needed."""
        return []

    def run(self):
        df = pd.read_csv(self.raw_path)
        clean_df, rules = self.clean(df)
        os.makedirs(CLEAN_DIR, exist_ok=True)
        clean_df.to_csv(self.clean_path, index=False)

        print(f"Rows: {len(clean_df)}")
        for line in self.summary_lines(clean_df):
            print(line)
        print()
        for name, count in rules:
            print(f"- {name}: {count}")
        return clean_df, rules
