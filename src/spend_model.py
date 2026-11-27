"""spend_model.py -- Part C task 3: collinearity check + Ridge/Lasso spend model + stability experiment.
Run:  python src/spend_model.py
Reads data/clean/subscribers.csv (Part A).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from constants import RANDOM_STATE, SUBSCRIBERS_CLEAN_PATH

CORRELATION_COLUMNS = ["hours_watched_per_week", "hd_hours_per_week", "sports_hours_per_week", "channels_watched"]
SPEND_FEATURES = [
    "monthly_recharge_inr", "tenure_months", "hours_watched_per_week", "hd_hours_per_week",
    "sports_hours_per_week", "channels_watched", "support_calls_90d",
]
TARGET = "next_cycle_recharge_inr"
ALPHA_GRID = {"model__alpha": [0.001, 0.01, 0.1, 0.5, 1, 2, 3, 10, 30, 100]}


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def report_correlation(df):
    section("Task 3a: collinearity")
    corr = df[CORRELATION_COLUMNS].corr()
    print(corr.round(3).to_string())
    print("\nPairs with |correlation| > 0.9:")
    for i, c1 in enumerate(CORRELATION_COLUMNS):
        for c2 in CORRELATION_COLUMNS[i + 1:]:
            value = corr.loc[c1, c2]
            if abs(value) > 0.9:
                print(f"  {c1} -- {c2}: {value:.3f}")
    return corr


def build_spend_splits(df):
    stay_df = df[df["churned"] == 0]
    X = stay_df[SPEND_FEATURES]
    y = stay_df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def fit_and_report(name, model, X_train, X_test, y_train, y_test, tune=False):
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    if tune:
        search = GridSearchCV(pipe, ALPHA_GRID, cv=5, scoring="r2")
        search.fit(X_train, y_train)
        best_pipe, best_alpha, cv_r2 = search.best_estimator_, search.best_params_["model__alpha"], search.best_score_
    else:
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
        pipe.fit(X_train, y_train)
        best_pipe, best_alpha = pipe, None

    pred = best_pipe.predict(X_test)
    test_r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5

    alpha_text = f"best alpha={best_alpha}, " if best_alpha is not None else ""
    print(f"{name}: {alpha_text}CV R2={cv_r2:.3f}, test R2={test_r2:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    return best_pipe


def report_lasso_zero_coefficients(lasso_pipe):
    section("Task 3c: Lasso zero coefficients")
    coefs = pd.Series(lasso_pipe.named_steps["model"].coef_, index=SPEND_FEATURES)
    print(coefs.round(4).to_string())
    zeroed = coefs[coefs == 0].index.tolist()
    print(f"\nFeatures with exactly zero coefficient: {zeroed}")
    print("Among a group of highly correlated, largely redundant features, keeping just one is enough to carry "
          "the shared signal, and Lasso's L1 penalty makes it cheaper overall to drive the others' coefficients "
          "to exactly zero rather than spread a similar-sized coefficient across every member of the group.")
    return zeroed


def stability_experiment(X_train, y_train):
    section("Task 3d: stability experiment")
    feature_idx = SPEND_FEATURES.index("hours_watched_per_week")
    linear_coefs, ridge_coefs = [], []
    for k in range(5):
        sample_X = X_train.sample(frac=0.8, random_state=k)
        sample_y = y_train.loc[sample_X.index]

        linear_pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        linear_pipe.fit(sample_X, sample_y)
        linear_coefs.append(linear_pipe.named_steps["model"].coef_[feature_idx])

        ridge_pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=30))])
        ridge_pipe.fit(sample_X, sample_y)
        ridge_coefs.append(ridge_pipe.named_steps["model"].coef_[feature_idx])

    linear_std = np.std(linear_coefs, ddof=0)
    ridge_std = np.std(ridge_coefs, ddof=0)
    print("hours_watched_per_week coefficient across 5 80% subsamples:")
    print(f"  LinearRegression: {[f'{c:.3f}' for c in linear_coefs]} -> population std={linear_std:.3f}")
    print(f"  Ridge(alpha=30):  {[f'{c:.3f}' for c in ridge_coefs]} -> population std={ridge_std:.3f}")
    print("When features are strongly correlated, plain linear regression can swap weight almost arbitrarily "
          "between them from one sample to the next, so an individual coefficient (and even its sign) should "
          "not be trusted as a stable business reading; Ridge's penalty shrinks and shares weight across the "
          "correlated group, giving a noticeably more stable, more trustworthy coefficient.")
    return linear_std, ridge_std


def build_best_ridge_pipeline(df=None):
    """Returns a fitted StandardScaler+Ridge pipeline (tuned alpha) on the churned=0 training split,
    for reuse by revenue_at_risk.py."""
    if df is None:
        df = pd.read_csv(SUBSCRIBERS_CLEAN_PATH)
    X_train, _X_test, y_train, _y_test = build_spend_splits(df)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    search = GridSearchCV(pipe, ALPHA_GRID, cv=5, scoring="r2")
    search.fit(X_train, y_train)
    return search.best_estimator_


def main():
    df = pd.read_csv(SUBSCRIBERS_CLEAN_PATH)
    report_correlation(df)

    X_train, X_test, y_train, y_test = build_spend_splits(df)
    print(f"\nTrain rows: {len(X_train)}, test rows: {len(X_test)}")

    section("Task 3b: spend model (LinearRegression, Ridge, Lasso)")
    fit_and_report("LinearRegression", LinearRegression(), X_train, X_test, y_train, y_test, tune=False)
    ridge_pipe = fit_and_report("Ridge", Ridge(), X_train, X_test, y_train, y_test, tune=True)
    lasso_pipe = fit_and_report("Lasso", Lasso(max_iter=10000), X_train, X_test, y_train, y_test, tune=True)

    report_lasso_zero_coefficients(lasso_pipe)
    stability_experiment(X_train, y_train)


if __name__ == "__main__":
    main()
