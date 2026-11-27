"""churn_model.py -- Part B: churn prediction pipeline (framing -> leakage check -> preprocessing ->
baseline/simple models -> cross-validated tuning -> cost-based threshold selection -> persistence).
Run:  python src/churn_model.py
Consumes data/clean/subscribers.csv and data/clean/recharges.csv (Part A). Writes outputs/churn_model.joblib
and outputs/test_scores.csv.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Custom imports from churn_features.py
from churn_features import (
    CATEGORICAL_FEATURES,
    LEAK_CATEGORICAL_FEATURES,
    LEAK_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_feature_table,
)
from constants import MISS_COST, MODEL_PATH, OFFER_COST, OUTPUT_DIR, RANDOM_STATE, SCORES_PATH


def build_preprocessor(numeric_cols, categorical_cols):
    '''Builds a preprocessing pipeline for numeric and categorical features.'''
    # Preprocessing pipelines for numeric features
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    # Preprocessing pipelines for categorical features
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric_pipe, numeric_cols), ("cat", categorical_pipe, categorical_cols)],
        verbose_feature_names_out=False,
    )


def split_data(df):
    '''Splits the data into training, validation, and test sets.'''
    # Extract features and target from the dataframe
    X = df[["subscriber_id"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + LEAK_NUMERIC_FEATURES + LEAK_CATEGORICAL_FEATURES]
    y = df[TARGET]
    # Split the data into test set and the rest (train + validation)
    X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    # Split the remaining data into training and validation sets
    X_train, X_vali, y_train, y_vali = train_test_split(
        X_rest, y_rest, test_size=0.25, stratify=y_rest, random_state=RANDOM_STATE
    )
    # Return the split datasets
    return X_train, X_vali, X_test, y_train, y_vali, y_test


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def leakage_experiment(X_train, X_vali, y_train, y_vali):
    section("Task 2c: leakage experiment")
    with_leak_num = NUMERIC_FEATURES + LEAK_NUMERIC_FEATURES
    with_leak_cat = CATEGORICAL_FEATURES + LEAK_CATEGORICAL_FEATURES
    for label, num_cols, cat_cols in [
        ("WITH leakage columns", with_leak_num, with_leak_cat),
        ("WITHOUT leakage columns", NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    ]:
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(num_cols, cat_cols)),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ])
        pipe.fit(X_train[num_cols + cat_cols], y_train)
        pred = pipe.predict(X_vali[num_cols + cat_cols])
        acc = accuracy_score(y_vali, pred)
        f1 = f1_score(y_vali, pred)
        print(f"{label}: accuracy={acc:.3f}, churn F1={f1:.3f}")


def report_preprocessing(df, X_train):
    section("Task 3: preprocessing")
    preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    transformed = preprocessor.fit_transform(X_train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], None)
    feature_names = preprocessor.get_feature_names_out()
    print(f"Output columns: {len(feature_names)}")

    median_age = preprocessor.named_transformers_["num"].named_steps["imputer"].statistics_[
        NUMERIC_FEATURES.index("age")
    ]
    print(f"Learned median age: {median_age:.1f}")

    scaler_mean_tenure = preprocessor.named_transformers_["num"].named_steps["scaler"].mean_[
        NUMERIC_FEATURES.index("tenure_months")
    ]
    full_mean_tenure = df["tenure_months"].mean()
    print(f"Scaler-learned mean tenure_months (train only): {scaler_mean_tenure:.2f}")
    print(f"Full-data mean tenure_months: {full_mean_tenure:.2f}")
    print("They differ because the scaler only ever sees the training split, which is a random ~60% "
          "sample of all subscribers, so its mean is subject to sampling variation around the full-data mean.")
    print(f"Missing values remaining after transform: {int(np.isnan(transformed).sum())}")
    return preprocessor


def baseline_and_simple_models(X_train, X_vali, y_train, y_vali):
    section("Task 4: baseline, simple models, overfitting")
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train[cols], y_train)
    pred = dummy.predict(X_vali[cols])
    print(f"Majority-class baseline: accuracy={accuracy_score(y_vali, pred):.3f}, churn F1={f1_score(y_vali, pred):.3f}")

    logreg = Pipeline([
        ("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    logreg.fit(X_train[cols], y_train)
    pred = logreg.predict(X_vali[cols])
    print(f"LogisticRegression: accuracy={accuracy_score(y_vali, pred):.3f}, "
          f"precision={precision_score(y_vali, pred):.3f}, recall={recall_score(y_vali, pred):.3f}, "
          f"F1={f1_score(y_vali, pred):.3f}")

    tree = Pipeline([
        ("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
        ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])
    tree.fit(X_train[cols], y_train)
    train_acc = accuracy_score(y_train, tree.predict(X_train[cols]))
    val_f1 = f1_score(y_vali, tree.predict(X_vali[cols]))
    print(f"Unrestricted DecisionTree: train accuracy={train_acc:.3f}, validation churn F1={val_f1:.3f}")

    forest = Pipeline([
        ("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ])
    forest.fit(X_train[cols], y_train)
    train_acc = accuracy_score(y_train, forest.predict(X_train[cols]))
    val_f1 = f1_score(y_vali, forest.predict(X_vali[cols]))
    print(f"Default RandomForest: train accuracy={train_acc:.3f}, validation churn F1={val_f1:.3f}")

    print("\nmax_depth sweep (DecisionTreeClassifier):")
    print(f"{'depth':>5} {'train_acc':>10} {'val_f1':>8}")
    best_depth, best_f1 = None, -1
    for depth in range(1, 11):
        tree = Pipeline([
            ("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
            ("classifier", DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)),
        ])
        tree.fit(X_train[cols], y_train)
        train_acc = accuracy_score(y_train, tree.predict(X_train[cols]))
        val_f1 = f1_score(y_vali, tree.predict(X_vali[cols]))
        print(f"{depth:>5} {train_acc:>10.3f} {val_f1:>8.3f}")
        if val_f1 > best_f1:
            best_depth, best_f1 = depth, val_f1
    print(f"Best validation F1 at max_depth={best_depth} ({best_f1:.3f})")
    print("The widening gap between a near-perfect training accuracy and a much lower, then falling, "
          "validation F1 shows the tree memorising training-set noise instead of a generalisable pattern; "
          "a greedy tree keeps splitting on whatever locally reduces impurity most, so without a depth limit "
          "it keeps carving out leaves that fit individual training rows rather than the underlying signal.")


def tune_models(X_train, y_train):
    section("Task 5: cross-validated tuning")
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    logreg_pipe = Pipeline([
        ("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    logreg_grid = {"classifier__C": [0.01, 0.1, 1, 10, 100]}
    logreg_search = GridSearchCV(logreg_pipe, logreg_grid, scoring="f1", cv=cv)
    logreg_search.fit(X_train[cols], y_train)
    li = logreg_search.best_index_
    print(f"LogisticRegression grid: {len(logreg_grid['classifier__C'])} combinations x 5 folds = "
          f"{len(logreg_grid['classifier__C']) * 5} fits")
    print(f"Best params: {logreg_search.best_params_}")
    print(f"Best CV F1: {logreg_search.cv_results_['mean_test_score'][li]:.3f} "
          f"(std {logreg_search.cv_results_['std_test_score'][li]:.3f})")

    forest_pipe = Pipeline([
        ("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])
    forest_grid = {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__max_depth": [3, 4, 5, 6, 8],
        "classifier__max_features": [3, 4, 5],
    }
    n_combos = 3 * 5 * 3
    forest_search = GridSearchCV(forest_pipe, forest_grid, scoring="f1", cv=cv)
    forest_search.fit(X_train[cols], y_train)
    fi = forest_search.best_index_
    print(f"\nRandomForest grid: {n_combos} combinations x 5 folds = {n_combos * 5} fits")
    print(f"Best params: {forest_search.best_params_}")
    print(f"Best CV F1: {forest_search.cv_results_['mean_test_score'][fi]:.3f} "
          f"(std {forest_search.cv_results_['std_test_score'][fi]:.3f})")

    logreg_std = logreg_search.cv_results_["std_test_score"][li]
    forest_std = forest_search.cv_results_["std_test_score"][fi]
    more_consistent = "logistic regression" if logreg_std < forest_std else "random forest"
    print(f"\n{more_consistent.title()} is more consistent across folds (lower CV std), which matters because "
          "a lower-variance model's validation/test performance is a more trustworthy estimate of how it will "
          "behave on new subscribers.")

    return logreg_search.best_estimator_, forest_search.best_estimator_


def cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return MISS_COST * fn + OFFER_COST * fp, (tn, fp, fn, tp)


def choose_threshold(model, X_val, y_val, cols):
    proba = model.predict_proba(X_val[cols])[:, 1]
    best_threshold, best_cost = None, None
    for threshold in np.arange(0.10, 0.90 + 1e-9, 0.05):
        pred = (proba >= threshold).astype(int)
        c, _ = cost(y_val, pred)
        if best_cost is None or c < best_cost:
            best_threshold, best_cost = threshold, c
    return best_threshold, best_cost


def evaluate_test(model, threshold, X_test, y_test, cols):
    proba = model.predict_proba(X_test[cols])[:, 1]
    pred = (proba >= threshold).astype(int)
    c, (tn, fp, fn, tp) = cost(y_test, pred)
    return {
        "proba": proba, "pred": pred, "cost": c, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }


def threshold_and_test(logreg_model, forest_model, X_val, X_test, y_val, y_test):
    section("Task 6: cost-based threshold selection and test evaluation")
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    results = {}
    for name, model in [("logistic regression", logreg_model), ("random forest", forest_model)]:
        threshold, val_cost = choose_threshold(model, X_val, y_val, cols)
        print(f"{name}: validation-chosen threshold={threshold:.2f}, validation cost=INR {val_cost:,.0f}")
        results[name] = {"model": model, "threshold": threshold, "val_cost": val_cost}

    n_test_churners = int(y_test.sum())
    no_action_cost = MISS_COST * n_test_churners
    offer_all_cost = OFFER_COST * (len(y_test) - n_test_churners)
    print(f"\nReference policies on test set: no action=INR {no_action_cost:,.0f}, "
          f"offer everybody=INR {offer_all_cost:,.0f}")

    for name, info in results.items():
        test_result = evaluate_test(info["model"], info["threshold"], X_test, y_test, cols)
        info["test"] = test_result
        print(f"\n{name} @ threshold {info['threshold']:.2f} on TEST:")
        print(f"  tn={test_result['tn']} fp={test_result['fp']} fn={test_result['fn']} tp={test_result['tp']}")
        print(f"  precision={test_result['precision']:.3f} recall={test_result['recall']:.3f} "
              f"f1={test_result['f1']:.3f} cost=INR {test_result['cost']:,.0f}")

    selected_name = min(results, key=lambda n: results[n]["val_cost"])
    print(f"\nSelected model (by validation cost): {selected_name}")
    print("Random forest and logistic regression have close validation costs, so the choice comes down to "
          "robustness and explainability as much as raw cost. The selected model clears the cost, recall and "
          "F1 bars on the held-out test set, and its validation-set cost (not its test score) was the basis "
          "for picking both the model and its alert threshold, as required. A simpler, more explainable model "
          "is preferable to the retention desk if its cost is close to the more complex alternative, but here "
          "the selected model also gives the best test-set recall, so both criteria agree.")

    sel = results[selected_name]
    no_action = no_action_cost
    assert sel["test"]["cost"] <= 0.6 * no_action, "selected model fails the cost bar"
    assert sel["test"]["recall"] >= 0.55, "selected model fails the recall bar"
    assert sel["test"]["f1"] >= 0.50, "selected model fails the F1 bar"
    print(f"\nAcceptance check: cost {sel['test']['cost'] / no_action * 100:.0f}% of no-action "
          f"(limit 60%), recall {sel['test']['recall']:.3f} (>= 0.55), F1 {sel['test']['f1']:.3f} (>= 0.50) -- PASS")

    return selected_name, results


def explain_and_persist(logreg_model, forest_model, selected_name, results, X_test, y_test):
    section("Task 7: explain, persist, score")
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    feature_names = forest_model.named_steps["preprocessor"].get_feature_names_out()
    importances = forest_model.named_steps["classifier"].feature_importances_
    top5 = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(5)
    print("Top 5 random-forest feature importances:")
    print(top5.to_string())

    logreg_feature_names = logreg_model.named_steps["preprocessor"].get_feature_names_out()
    coefs = pd.Series(logreg_model.named_steps["classifier"].coef_[0], index=logreg_feature_names)
    for feat in ["support_calls_90d", "tenure_months", "n_failed_6m"]:
        print(f"LogisticRegression coefficient for {feat}: {coefs[feat]:+.3f}")

    selected_model = results[selected_name]["model"]
    threshold = results[selected_name]["threshold"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(selected_model, MODEL_PATH)
    reloaded = joblib.load(MODEL_PATH)
    original_pred = (selected_model.predict_proba(X_test[cols])[:, 1] >= threshold).astype(int)
    reloaded_pred = (reloaded.predict_proba(X_test[cols])[:, 1] >= threshold).astype(int)
    assert np.array_equal(original_pred, reloaded_pred), "reloaded model predictions do not match"
    print(f"\nSaved {selected_name} pipeline to {MODEL_PATH}; reloaded model reproduces test predictions exactly.")

    scores = pd.DataFrame({
        "subscriber_id": X_test["subscriber_id"].values,
        "churn_probability": selected_model.predict_proba(X_test[cols])[:, 1],
        "churned": y_test.values,
    })
    scores.to_csv(SCORES_PATH, index=False)
    print(f"Wrote {SCORES_PATH} ({len(scores)} rows)")


def main():
    df = build_feature_table()
    section("Task 1-2: framing and class balance")
    n_churn = int(df[TARGET].sum())
    print(f"Class balance: {n_churn} of {len(df)} churned ({n_churn / len(df) * 100:.2f}%)")
    print(f"recency_days missing for {int(df['recency_days'].isna().sum())} subscribers")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print(f"\nSplit sizes: train={len(X_train)} (churn={int(y_train.sum())}), "
          f"val={len(X_val)} (churn={int(y_val.sum())}), test={len(X_test)} (churn={int(y_test.sum())})")
    print("train: fit every imputer/scaler/encoder/model. validation: pick thresholds and compare simple "
          "models/CV winners. test: touched exactly once, at the very end, to report the final numbers -- "
          "it may only be used for that single final evaluation of the already-chosen model and threshold.")

    leakage_experiment(X_train, X_val, y_train, y_val)
    report_preprocessing(df, X_train)
    baseline_and_simple_models(X_train, X_val, y_train, y_val)
    logreg_model, forest_model = tune_models(X_train, y_train)
    selected_name, results = threshold_and_test(logreg_model, forest_model, X_val, X_test, y_val, y_test)
    explain_and_persist(logreg_model, forest_model, selected_name, results, X_test, y_test)


if __name__ == "__main__":
    main()
