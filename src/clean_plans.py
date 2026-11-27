"""clean_plans.py -- copies data/raw/plans.csv to data/clean/plans.csv unchanged (no cleaning needed).
Run:  python src/clean_plans.py
"""
from data_cleaner import BaseCleaner


class PlansCleaner(BaseCleaner):
    def __init__(self):
        super().__init__("plans.csv", "plans.csv")

    def clean(self, df):
        return df, [("Copied unchanged", 0)]


if __name__ == "__main__":
    PlansCleaner().run()
