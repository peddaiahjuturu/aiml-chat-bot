# AIML Chat Bot -- Tata Play Subscriber Retention & Support Copilot (Capstone)

Synthetic-data pipeline and offline mock LLM for a subscriber retention / support chatbot capstone project.

## Project structure

```
generate_data.py        # creates all synthetic raw data + policy docs (stdlib only)
mock_llm.py              # offline, deterministic stand-in for a hosted chat-completion API
src/
  audit_data.py          # reports shape/dtypes/missing values + data-quality figures for the raw tables
  data_cleaner.py        # shared BaseCleaner class used by the three clean_*.py scripts
  clean_subscribers.py   # cleans data/raw/subscribers.csv -> data/clean/subscribers.csv
  clean_recharges.py     # cleans data/raw/recharges.csv  -> data/clean/recharges.csv
  clean_plans.py         # copies data/raw/plans.csv -> data/clean/plans.csv unchanged
  store_db.py            # writes the three clean tables to data/tataplay.db (SQLite)
  run_queries.py         # runs sql/queries.sql against data/tataplay.db and prints results
  churn_features.py      # builds the Part B churn modelling table (clean subscribers + recharge features)
  churn_model.py         # Part B: split, leakage check, preprocessing, tuning, threshold selection, persistence
  constants.py           # shared paths and parameters used across the scripts above
  segments.py            # Part C: K-means subscriber segmentation -> outputs/segments.csv
  spend_model.py         # Part C: collinearity check + Ridge/Lasso next-cycle-recharge spend model
  revenue_at_risk.py     # Part C: revenue-at-risk priority list -> outputs/retention_priority.csv
  llm_client.py          # Part D: reliable call_llm() client (mock by default, live via LLM_MODE=live)
  llm_client_demo.py     # Part D: demonstrates 401/404/length/no-memory/rate-limit-retry behaviour
  ticket_classifier.py   # Part D: weak/zero-shot/few-shot ticket classification -> outputs/ticket_classification.csv
  ticket_extraction.py   # Part D: validated TicketRecord extraction with retry -> outputs/ticket_extractions.jsonl
  policy_rag.py          # Part D: chunking + ChromaDB retrieval + grounded answer() -> outputs/rag_eval.csv
  copilot.py             # Part D: Retention Copilot build_brief() -> outputs/retention_briefs.jsonl
sql/
  queries.sql            # the four analysis queries (Q1-Q4)
data/
  raw/                   # generated, uncleaned source tables
  clean/                 # cleaned tables produced by the src/clean_*.py scripts
  policies/              # policy text documents used by the retention/support copilot
  tataplay.db             # SQLite database holding the clean subscribers/recharges/plans tables
outputs/
  churn_model.joblib      # persisted, fitted churn-prediction pipeline (selected model)
  test_scores.csv         # per-test-subscriber churn probability + actual churned label
  segments.csv            # subscriber_id, segment_id, segment_name for all 2,400 clean subscribers
  retention_priority.csv  # top-15 test subscribers ranked by revenue at risk
  ticket_classification.csv  # ticket_id, true_category, zero_shot, few_shot for the 146 readable tickets
  ticket_extractions.jsonl   # one line per ticket (150): ticket_id, status, attempts, record
  rag_eval.csv               # the 16 policy_eval.csv questions: retrieval + grounded/ungrounded answers
  retention_briefs.jsonl     # one validated retention brief per priority-list subscriber (15)
```

## Requirements

Minimum packages (see [requirements.txt](requirements.txt)):

| Package | Purpose |
|---|---|
| `pandas` | Reading/cleaning/auditing the CSV tables (core data-processing library used throughout `src/`) |
| `numpy` | Numerical operations, required by `pandas` |
| `requests` | HTTP client, used to call chat-completion APIs with the same shape as `mock_llm.py` |
| `python-dotenv` | Loads API keys/config from a local `.env` file |
| `python-dateutil` | Date parsing support used by `pandas` |
| `tzdata` | Timezone database required by `pandas`/`python-dateutil` on some platforms |
| `certifi`, `charset-normalizer`, `idna`, `urllib3` | Transitive dependencies of `requests` (TLS certificates, encoding detection, URL handling) |

`generate_data.py` and `mock_llm.py` use only the Python standard library and need no extra packages.

## Setup (for anyone cloning this repo)

1. **Create a virtual environment** (from the repo root):
   ```powershell
   python -m venv .venv
   ```
2. **Activate it:**
   - PowerShell (Windows):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - Command Prompt (Windows):
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Generate the synthetic raw data** (writes `data/raw/*.csv` and `data/policies/*.txt`):
   ```powershell
   python generate_data.py
   ```
   Output:
   ```text
   generated: subscribers=2415 recharges_raw=13055 tickets=150 policy_files=5
   ```

   > **Do not use these two columns as features:** `closure_request_logged` and `next_cycle_recharge_inr`. They are not known at prediction time (they are only logged/known once a subscriber is already leaving or has started their next billing cycle), so they must be excluded from any churn-prediction model to avoid data leakage.

## Part A -- Data Foundation: Audit, Clean, Store, Query

Everything downstream trusts this Part. The raw exports contain duplicates, inconsistent spellings, impossible values and text where numbers belong. This Part produces the clean tables and the SQLite database that Parts B, C and D read. Data is cleaned in pandas; no missing value is filled in this Part (imputation belongs inside the Part B pipeline, after the split, so that no test row can influence it).

## Steps completed so far

1. Generated synthetic raw data with `generate_data.py` (2,415 subscribers, 13,055 raw recharges, 150 tickets, and 5 policy documents).
2. Audited the raw tables with `src/audit_data.py` to understand shape, dtypes, missing values, duplicate rows, and spelling/id inconsistencies before cleaning.
3. Cleaned `subscribers.csv`, `recharges.csv` and `plans.csv` with `src/clean_subscribers.py`, `src/clean_recharges.py` and `src/clean_plans.py`, writing the results to `data/clean/`.
4. Stored the three clean tables in a SQLite database (`data/tataplay.db`) with `src/store_db.py`, and ran four analysis queries (`sql/queries.sql`) against it with `src/run_queries.py`.

### Audit results (`python src/audit_data.py`, run against `data/raw/*.csv`)

**`data/raw/subscribers.csv`** -- 2,415 rows x 17 columns

| Column | dtype | missing_count | missing_pct |
|---|---|---|---|
| subscriber_id | str | 0 | 0.00 |
| city | str | 0 | 0.00 |
| plan_id | str | 0 | 0.00 |
| tenure_months | int64 | 0 | 0.00 |
| age | float64 | 99 | 4.10 |
| payment_mode | str | 116 | 4.80 |
| auto_renew | str | 0 | 0.00 |
| add_on_count | int64 | 0 | 0.00 |
| monthly_recharge_inr | int64 | 0 | 0.00 |
| hours_watched_per_week | float64 | 0 | 0.00 |
| hd_hours_per_week | float64 | 0 | 0.00 |
| sports_hours_per_week | float64 | 0 | 0.00 |
| channels_watched | int64 | 0 | 0.00 |
| support_calls_90d | int64 | 0 | 0.00 |
| closure_request_logged | str | 0 | 0.00 |
| next_cycle_recharge_inr | float64 | 516 | 21.37 |
| churned | int64 | 0 | 0.00 |

- Missing `age` and `payment_mode` are genuine missing values (not yet imputed).
- `next_cycle_recharge_inr` is missing by design for churned subscribers (it's only known after their next billing cycle).
- Exact duplicate rows: **15**; distinct `subscriber_id`: **2,400**.
- `age` values outside the plausible 10-100 range: **6**.
- Distinct raw `city` spellings: **22**; distinct raw `auto_renew` spellings: **6**.

**`data/raw/recharges.csv`** -- 13,055 rows x 6 columns

| Column | dtype | missing_count | missing_pct |
|---|---|---|---|
| recharge_id | str | 0 | 0.0 |
| subscriber_id | str | 0 | 0.0 |
| recharge_date | str | 0 | 0.0 |
| amount_inr | str | 0 | 0.0 |
| status | str | 0 | 0.0 |
| channel | str | 0 | 0.0 |

- No missing values, but several columns contain "messy" raw values (mixed-case `status`, `amount_inr` sometimes stored as text like `"INR 299"`) that the cleaning scripts standardise.
- Exact duplicate rows: **60**; distinct `recharge_id`: **12,995**.
- `amount_inr` values that are not plain numbers: **366**.
- Distinct raw `status` spellings: **6**.
- Rows whose `subscriber_id` does not exist in `subscribers.csv`: **10**.

### Cleaning results

No missing value is ever filled in by the cleaning scripts -- cleaning only removes rows and standardises/corrects existing values (the `amount_inr` rebuild below replaces a malformed text value with the same subscriber's known `monthly_recharge_inr`, it does not fill a blank).

**`python src/clean_subscribers.py`** (`data/raw/subscribers.csv` -> `data/clean/subscribers.csv`)

- Output rows: **2,400** (started from 2,415)
- City counts: Mumbai 524, Delhi 436, Chennai 391, Bengaluru 386, Hyderabad 262, Pune 245, Kolkata 156
- `auto_renew` counts: No 1,355, Yes 1,045
- `age` still missing: 104; `payment_mode` still missing: 114
- Churned subscribers: 514 (21.42%)
- Rules applied:
  - Removed exact duplicate rows: 15
  - Standardised city to one of 7 canonical names: 892
  - Standardised `auto_renew` to Yes/No: 695
  - Set `age` to missing where outside 10-100: 6

**`python src/clean_recharges.py`** (`data/raw/recharges.csv` -> `data/clean/recharges.csv`)

- Output rows: **12,985** (started from 13,055)
- `status` counts: success 11,442, failed 1,422, pending 121
- Total successful recharge amount: **INR 6,457,573**
- Rules applied, in order:
  - Removed exact duplicate rows: 60
  - Removed recharges for unknown subscribers (not present in cleaned subscribers table): 10
  - Lower-cased `status`: 5,099
  - Converted `amount_inr` to a number; where it could not be converted (e.g. `"INR 498"`), rebuilt it as that subscriber's `monthly_recharge_inr`: 364
  - Parsed `recharge_date` to a date type (unparsable values set aside): 0

**`python src/clean_plans.py`** (`data/raw/plans.csv` -> `data/clean/plans.csv`)

- Copied unchanged (6 rows, no cleaning needed).

## Store and query

`src/store_db.py` writes the three clean tables (`subscribers`, `recharges`, `plans`) to `data/tataplay.db` using `sqlite3` + `DataFrame.to_sql`. `sql/queries.sql` holds four analysis queries, run from Python with `src/run_queries.py` (uses `sqlite3` + `pandas.read_sql_query`). Unless stated otherwise, every query uses only `status = 'success'` recharges.

**Q1 -- revenue by plan** (`INNER JOIN` recharges -> subscribers -> plans, grouped by `plan_name`, ordered by revenue):

| plan_name | total_revenue | avg_revenue | distinct_subscribers |
|---|---|---|---|
| Premium | 1,471,199 | 860.35 | 339 |
| Standard | 1,323,622 | 492.60 | 559 |
| Standard HD | 1,283,471 | 645.61 | 407 |
| Premium Plus | 1,262,398 | 1,075.30 | 228 |
| Basic | 782,554 | 326.61 | 493 |
| Basic Lite | 334,329 | 224.83 | 298 |

Cross-checked with a pandas `groupby("plan_name")["amount_inr"].agg(["sum","mean"])` on the merged recharges/subscribers/plans frame -- matches the SQL result.

**Q2 -- subscribers who never had a successful recharge** (`LEFT JOIN` subscribers to successful recharges, keep rows where the recharge key `IS NULL`):

- **76** subscribers, of whom **68** have `tenure_months = 0`.
- An `INNER JOIN` would silently hide these 76 rows: it only keeps subscribers that have at least one matching successful-recharge row, so a subscriber with zero successful recharges has nothing to match and is dropped from the result entirely instead of appearing with a `NULL` recharge.

**Q3 -- ranking within a city** (`RANK()`, `DENSE_RANK()`, `ROW_NUMBER()` over each subscriber's total successful recharge amount, `PARTITION BY city ORDER BY total DESC`, `ROW_NUMBER` ties broken by `subscriber_id`):

Subscribers with `RANK() = 1` per city: Bengaluru 10, Chennai 10, Delhi 5, Hyderabad 5, Kolkata 2, Mumbai 6, Pune 4. `ROW_NUMBER() = 1` is always exactly 1 per city.

Kolkata's first four rows:

| subscriber_id | city | total_amount | rank_ | dense_rank_ | row_number_ |
|---|---|---|---|---|---|
| TP00074 | Kolkata | 6,876 | 1 | 1 | 1 |
| TP00917 | Kolkata | 6,876 | 1 | 1 | 2 |
| TP00143 | Kolkata | 6,582 | 3 | 2 | 3 |
| TP01355 | Kolkata | 6,582 | 3 | 2 | 4 |

For "the top spender per city", use **`RANK()`**: `DENSE_RANK()` would mark every tied subscriber as rank 1 with no gap, which is fine for ties but `ROW_NUMBER()` would arbitrarily pick only one of the tied subscribers as "the" top spender -- `RANK()` correctly reports every subscriber tied for first place while still reflecting that gaps exist below a tie.

**Q4 -- running total for subscriber `TP00010`** (`SUM(amount_inr) OVER (ORDER BY recharge_date, recharge_id)`):

| recharge_id | recharge_date | amount_inr | running_total |
|---|---|---|---|
| R000053 | 2025-10-15 | 498 | 498 |
| R000055 | 2025-12-15 | 498 | 996 |
| R000056 | 2026-01-15 | 498 | 1,494 |
| R000057 | 2026-02-15 | 498 | 1,992 |
| R000058 | 2026-03-13 | 498 | 2,490 |

5 rows; the running total ends at **2,490**.

## Part B -- Churn Prediction You Can Trust

Predicts which subscribers will churn, using `data/clean/subscribers.csv` and `data/clean/recharges.csv` from Part A. Run with `python src/churn_model.py` (requires `data/clean/*.csv` to already exist).

### 1. Framing

| Question | Answer |
|---|---|
| What decision does the prediction drive? | Whether the retention desk proactively offers a retention deal to a subscriber before their next billing cycle. |
| Who acts, and what do they do differently? | The retention desk; for subscribers flagged "will churn" they make a retention call / send a discount offer instead of doing nothing. |
| Exact target column | `churned` (from `data/clean/subscribers.csv`). |
| Is the target a number, a category, or absent? | A category -- binary (0 = stays, 1 = churns). |
| Cost of a wrong prediction | Missing a subscriber who then leaves costs **INR 1,200** (lost revenue); offering a deal to a subscriber who would have stayed costs **INR 300** (wasted discount). |

Class balance: **514 of 2,400 churned (21.42%)**.

### 2. Modelling table, split, and leakage

Each subscriber is joined to three recharge-derived features computed from the clean recharges as of the snapshot date **2026-03-31**: `n_success_6m`, `n_failed_6m` (counts of successful/failed recharges; 0 if the subscriber has no recharge rows), and `recency_days` (days since the last successful recharge; missing if none). `recency_days` is missing for **76 subscribers** -- the same 76 subscribers identified in Part A's Q2 as never having a successful recharge.

- **12 numeric features:** `tenure_months, age, add_on_count, monthly_recharge_inr, hours_watched_per_week, hd_hours_per_week, sports_hours_per_week, channels_watched, support_calls_90d, n_success_6m, n_failed_6m, recency_days`
- **4 categorical features:** `city, plan_id, payment_mode, auto_renew`

**Split** (`train_test_split`, test first, then train/val out of the rest, both `stratify=y, random_state=42`):

| Set | Rows | Churners |
|---|---|---|
| train | 1,440 | 308 |
| validation | 480 | 103 |
| test | 480 | 103 |

`train` fits every imputer/scaler/encoder/model. `validation` is used to pick alert thresholds and to compare simple models and the cross-validated winners. `test` is touched exactly once, at the very end, for the single final report of the already-chosen model and threshold -- it is never used to fit, tune, or pick a threshold.

**Leakage columns:** per the data dictionary's *Known when* column, `closure_request_logged` and `next_cycle_recharge_inr` are not known at prediction time (they're only logged once a subscriber is already leaving, or once their next billing cycle has happened) and must never be model features.

Proof of the damage -- `RandomForestClassifier(n_estimators=100, random_state=42)` on the validation set:

| Features | Accuracy | Churn F1 |
|---|---|---|
| **With** the two leakage columns | 0.983 | 0.961 |
| **Without** the two leakage columns | 0.823 | 0.514 |

A model that scores 0.983/0.961 by reading `closure_request_logged` is not shippable: that column is only written down once the cancellation process has already started, so in production, at the moment a prediction is actually needed, the column does not exist yet -- the "leaky" score measures how well the model can read an answer key, not how well it predicts the future.

### 3. Preprocessing

One `ColumnTransformer`, fit on the **training rows only**: numeric columns get `SimpleImputer(strategy="median")` then `StandardScaler`; categorical columns get `SimpleImputer(strategy="most_frequent")` then `OneHotEncoder(handle_unknown="ignore")`. Every later model is a `Pipeline` containing this transformer, so it is refit from scratch on whatever training fold that model sees -- it never sees validation or test rows.

- Output columns: **31**
- Learned median `age`: **38.0**
- Scaler-learned mean `tenure_months` (train only): **22.65**, vs. full-data mean **22.46** -- they differ because the scaler only ever sees the training split, a random ~60% sample of all subscribers, so its mean carries ordinary sampling variation around the full-data mean.
- Missing values remaining after transform: **0** (confirmed in code).

### 4. Baseline, simple models, and overfitting

Validation-set results:

| Model | Accuracy | Precision | Recall | Churn F1 |
|---|---|---|---|---|
| Majority-class baseline | 0.785 | -- | 0 | 0.000 |
| LogisticRegression(max_iter=1000) | 0.833 | 0.646 | 0.495 | 0.560 |

| Model | Train accuracy | Validation churn F1 |
|---|---|---|
| Unrestricted DecisionTreeClassifier | 1.000 | 0.459 |
| Default RandomForestClassifier(n_estimators=100) | 1.000 | 0.514 |

`max_depth` sweep (DecisionTreeClassifier, train accuracy vs. validation churn F1):

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| train_acc | 0.815 | 0.832 | 0.839 | 0.844 | 0.863 | 0.883 | 0.908 | 0.924 | 0.944 | 0.957 |
| val_f1 | 0.434 | 0.413 | 0.408 | 0.422 | 0.529 | **0.545** | 0.488 | 0.489 | 0.468 | 0.440 |

Best validation F1 is at `max_depth=6` (0.545). The widening gap between near-perfect training accuracy and a much lower (then falling) validation F1 shows the tree memorising training-set noise rather than a generalisable pattern; a greedy tree keeps splitting on whatever locally reduces impurity most, so without a depth limit it keeps carving out leaves that fit individual training rows instead of the underlying signal.

### 5. Cross-validated tuning

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` + `GridSearchCV(scoring="f1")`, fit on the **training rows only**.

| Model | Grid | Combinations | Fits | Best params | Best CV F1 (std) |
|---|---|---|---|---|---|
| LogisticRegression | `C in {0.01, 0.1, 1, 10, 100}` | 5 | 25 | `C=100` | 0.507 (0.035) |
| RandomForestClassifier | `n_estimators in {50,100,150}`, `max_depth in {3,4,5,6,8}`, `max_features in {3,4,5}` | 45 | 225 | `max_depth=8, max_features=5, n_estimators=100` | 0.439 (0.061) |

Logistic regression is more consistent across folds (CV std 0.035 vs. 0.061 for the random forest), which matters because a lower-variance model's validation/test performance is a more trustworthy estimate of how it will behave on new subscribers.

### 6. Cost-based threshold selection and test-set evaluation

For each tuned model, every threshold from 0.10 to 0.90 (step 0.05) is tried against its validation-set `predict_proba`; cost = `1,200 x missed churners + 300 x needless offers`. The cheapest threshold per model, chosen on validation:

| Model | Threshold | Validation cost |
|---|---|---|
| Random forest | 0.30 | INR 53,100 |
| Logistic regression | 0.20 | INR 56,400 |

Reference policies on the test set: **no action** = INR 123,600 (1,200 x 103 test churners); **offer everybody** = INR 113,100 (300 x 377 non-churners).

Applying each model with its validation-chosen threshold to the **test** set (touched here for the first and only time):

| Model | tn | fp | fn | tp | Precision | Recall | F1 | Cost |
|---|---|---|---|---|---|---|---|---|
| Random forest @ 0.30 | 310 | 67 | 29 | 74 | 0.525 | 0.718 | 0.607 | INR 54,900 |
| Logistic regression @ 0.20 | 275 | 102 | 17 | 86 | 0.457 | 0.835 | 0.591 | INR 51,000 |

**Selected model: random forest** -- chosen on its lower validation cost (INR 53,100 vs. 56,400), not its test result. Random forest and logistic regression land close on cost, so the decision weighs robustness as much as raw cost; a tree ensemble's feature-importance ranking is also easy to explain to a non-technical retention desk ("more support calls -> higher risk"), similar to how logistic-regression coefficients would be explained, so explainability doesn't clearly favour either model here -- the validation cost breaks the tie.

Acceptance check on test: cost is **44%** of the no-action cost (limit 60%, i.e. <= INR 74,160) -- **pass**; recall **0.718** (>= 0.55) -- **pass**; F1 **0.607** (>= 0.50) -- **pass**.

### 7. Explain, persist, and score

Top 5 random-forest feature importances (`feature_importances_`, names from the pipeline's `get_feature_names_out()`):

| Feature | Importance | Business reading |
|---|---|---|
| `support_calls_90d` | 0.196 | More support calls in the last 90 days -> higher churn risk. |
| `recency_days` | 0.091 | The longer since the last successful recharge, the higher the churn risk. |
| `tenure_months` | 0.082 | Newer subscribers churn more; long-tenured subscribers are safer. |
| `n_success_6m` | 0.079 | Subscribers with fewer successful recharges recently are more likely to churn. |
| `hd_hours_per_week` | 0.067 | Lower HD viewing hours associate with higher churn risk. |

LogisticRegression coefficient signs: `support_calls_90d` **+1.075** (more calls -> more likely to churn), `tenure_months` **-0.511** (longer tenure -> less likely to churn), `n_failed_6m` **+0.211** (more failed recharges -> more likely to churn).

The selected (random forest) pipeline is saved with `joblib.dump` to `outputs/churn_model.joblib`; reloading it with `joblib.load` reproduces the exact same test-set predictions. `outputs/test_scores.csv` holds one row per test subscriber (480 rows) with columns `subscriber_id, churn_probability, churned`.

**No test-set leakage:** the `ColumnTransformer` (imputer/scaler/encoder), every model fit, every `GridSearchCV` cross-validation fold, and the cost-based threshold search all only ever see `train` and/or `validation` rows. The `test` set is loaded only once, at the very end, to compute the final confusion matrix/metrics/cost for the already-selected model and its already-chosen threshold -- in that order: **train -> validation -> test**.

## Part C -- Segments, Spend Model and Revenue at Risk

Reads `data/clean/subscribers.csv` (Part A) and `outputs/test_scores.csv` (Part B). Run with `python src/segments.py`, `python src/spend_model.py` and `python src/revenue_at_risk.py` (in that order, after Part B).

### 1. Cluster with K-means

`KMeans(n_clusters=K, n_init=10, random_state=42)` on `StandardScaler`-standardised `tenure_months`, `hours_watched_per_week`, `monthly_recharge_inr` (all 2,400 clean subscribers), swept over K=2..8:

| K | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| inertia | 3728.2 | 2813.0 | 2193.9 | 1781.1 | 1502.5 | 1292.8 | 1191.4 |
| silhouette | 0.420 | 0.346 | 0.340 | 0.343 | 0.344 | 0.364 | 0.336 |

Inertia always falls as K grows because adding another centroid can only reduce (never increase) the total within-cluster distance, so K=8 beats K=2 on inertia by construction. K=2's silhouette is the best here because splitting the data into two big, well-separated blobs is the easiest shape to score well on average, not because two segments are the most useful split for the business. Neither metric alone picks K: inertia always prefers more clusters and silhouette always prefers fewer, so the final choice needs the elbow (where inertia stops dropping quickly) plus whether the resulting groups are distinct and actionable. **Chosen K = 4.**

### 2. Profile, name and assign

| segment_id | size | mean tenure (months) | mean weekly hours | mean monthly recharge (INR) | churn rate | Name | Recommended action |
|---|---|---|---|---|---|---|---|
| 0 | 593 | 23.58 | 7.08 | 301 | 0.22 | Casual Viewer | Light-touch engagement nudges (content recommendations) to lift usage. |
| 1 | 543 | 34.16 | 22.24 | 560 | 0.13 | Loyal Core | Maintain with standard renewal communication; low-cost to retain. |
| 2 | 480 | 34.20 | 33.64 | 942 | 0.11 | **Premium** (highest mean monthly recharge) | White-glove support; protect this high-value, low-churn group. |
| 3 | 784 | 6.32 | 14.85 | 514 | 0.33 | **At-Risk** (highest churn rate) | Proactive retention outreach / discount offer soon after sign-up. |

Segment numbers (0-3) are arbitrary labels with no inherent order -- only the profile above determines the business name.

**New subscriber** (tenure 4 months, 15 hours/week, monthly recharge INR 520), classified by hand: scaled with the fitted `StandardScaler`, then the Euclidean distance to each of the 4 centroids was computed as `[1.708, 2.094, 3.117, 0.153]` -- nearest is segment 3 (**At-Risk**) at distance **0.153** (next nearest, Casual Viewer, at 1.708), and `kmeans.predict` agrees. `outputs/segments.csv` holds `subscriber_id, segment_id, segment_name` for all 2,400 subscribers.

### 3. Collinearity and a spend model (Ridge / Lasso)

Correlation matrix of the four viewing/channel columns -- pairs with |correlation| > 0.9:

| Pair | Correlation |
|---|---|
| `hours_watched_per_week` -- `hd_hours_per_week` | 0.979 |
| `hours_watched_per_week` -- `sports_hours_per_week` | 0.947 |
| `hours_watched_per_week` -- `channels_watched` | 0.938 |
| `hd_hours_per_week` -- `sports_hours_per_week` | 0.927 |
| `hd_hours_per_week` -- `channels_watched` | 0.921 |

(`sports_hours_per_week` -- `channels_watched` = 0.891 is below the 0.9 line.)

Predicting `next_cycle_recharge_inr` for subscribers with `churned = 0` from 7 features (`monthly_recharge_inr, tenure_months, hours_watched_per_week, hd_hours_per_week, sports_hours_per_week, channels_watched, support_calls_90d`), split `train_test_split(test_size=0.2, random_state=42)` into **1,508 train / 378 test** rows. Each is a `StandardScaler` + model pipeline; `Ridge`/`Lasso(max_iter=10000)` alphas tuned with `GridSearchCV(cv=5, scoring="r2")` over `{0.001, 0.01, 0.1, 0.5, 1, 2, 3, 10, 30, 100}`:

| Model | Best alpha | CV R² | Test R² | MAE | RMSE |
|---|---|---|---|---|---|
| LinearRegression | -- | 0.994 | 0.993 | 20.16 | 24.70 |
| Ridge | 0.01 | 0.994 | 0.993 | 20.16 | 24.70 |
| Lasso | 0.1 | 0.994 | 0.993 | 20.11 | 24.65 |

**Lasso's exactly-zero coefficients:** `tenure_months`, `hours_watched_per_week`, `channels_watched`. Among a group of highly correlated, largely redundant features, keeping just one (`hd_hours_per_week`, plus `sports_hours_per_week`) is enough to carry the shared signal, and Lasso's L1 penalty makes it cheaper overall to drive the others to exactly zero rather than spread a similar-sized coefficient across every member of the group.

**Stability experiment:** fitting `StandardScaler` + `LinearRegression` and `StandardScaler` + `Ridge(alpha=30)` on five random 80% subsamples of the training rows (`random_state=0..4`) and tracking the standardised `hours_watched_per_week` coefficient:

| Model | Coefficients across 5 subsamples | Population std (`ddof=0`) |
|---|---|---|
| LinearRegression | -5.731, -1.712, -4.839, -7.736, -4.733 | **1.945** |
| Ridge(alpha=30) | 13.260, 15.554, 13.581, 13.024, 13.786 | **0.896** |

When features are strongly correlated, plain linear regression can swap weight almost arbitrarily between them from one sample to the next, so an individual coefficient (and even its sign) should not be trusted as a stable business reading; Ridge's penalty shrinks and shares weight across the correlated group, giving a noticeably more stable, more trustworthy coefficient.

### 4. Revenue at risk

For the 480 Part B test subscribers, the tuned Ridge pipeline predicts `next_cycle_recharge_inr` for every subscriber (including those who churned), and `revenue_at_risk_inr = churn_probability x predicted_next_recharge_inr`. The top 15 by revenue at risk are written to `outputs/retention_priority.csv` (`subscriber_id, churn_probability, predicted_next_recharge_inr, revenue_at_risk_inr, segment_name, churned`):

- Sum of revenue at risk (top 15): **INR 8,717**
- **12 of 15** actually churned (bar: at least 9 -- **pass**; base churn rate is 21%)

This list is built from test-set subscribers whose outcome (`churned`) is already known -- it illustrates the revenue-at-risk method, it is **not** a forward-looking evaluation of unseen subscribers.

## Part D -- Support Copilot: Prompts, Validation, Retrieval, Retention Briefs

Built around the supplied offline `mock_llm.py`, using LangChain's prompt templates/parsers (`langchain_core`), Pydantic validation, and a ChromaDB vector store. **Embedding function used: `chromadb.utils.embedding_functions.ONNXMiniLM_L6_V2()`** (Chroma's built-in ONNX runtime for `all-MiniLM-L6-v2`), not `SentenceTransformerEmbeddingFunction`, to avoid the heavy PyTorch dependency -- it reproduces every reference number below.

### A. A reliable client for the chat API (`src/llm_client.py`)

`call_llm(messages, temperature=0.0, max_tokens=300)` builds the `Authorization: Bearer ...` / `Content-Type` headers and the `{"model": "openai/gpt-4o-mini", "messages", "temperature", "max_tokens"}` payload, and posts it to `mock_llm.post(mock_llm.URL, ...)` by default. It returns the reply text only on `200`; retries **only** on `429`, at most 3 times, sleeping `BASE_DELAY * 2**n` seconds before retry `n` (`n = 0, 1, 2`; `BASE_DELAY = 0.05`); raises `LLMError` immediately (no retry) for `400`, `401`, `404`; and prints a note when `finish_reason == "length"`.

**A real endpoint is reachable by configuration alone:** when the environment variable `LLM_MODE=live` is set, `call_llm` instead calls `requests.post` against the real `mock_llm.URL`, reading the key from the `OPENROUTER_API_KEY` environment variable -- the key is never hard-coded in the source. No live call is made during grading; the switch only changes which function (`mock_llm.post` vs. `requests.post`) is used to send the already-built request.

Demonstrations (`python src/llm_client_demo.py`), each after `mock_llm.reset()`:

| Demonstration | Result |
|---|---|
| (a) No `Authorization` header | status **401** |
| (a) Wrong URL | status **404** |
| (b) `max_tokens=3` | reply truncated, `finish_reason` **"length"** (printed note; reply `'I am'`) |
| (c) "What is my name?" alone | `"I do not know your name."` |
| (c) "My name is Asha." + assistant reply + "What is my name?" in the same messages list | `"Your name is Asha."` (no memory across *separate* calls) |
| (d) 30 simple calls in a loop | **33** total calls to `mock_llm.post` (3 rate-limited responses, each recovered by one retry) |

### B. Prompt design and few-shot classification (`src/ticket_classifier.py`)

Built with `ChatPromptTemplate.from_messages(...) | _call_model | StrOutputParser()`, where `_call_model` is a plain function that takes the prompt value, converts `.to_messages()` (`.type` in `system`/`human`/`ai`, `.content`) to the API's `{"role", "content"}` format, and calls `call_llm`. Run over the **146 readable tickets** (the 4 `...` tickets dropped), `mock_llm.reset()` before each prompt:

| Prompt | Valid labels | Accuracy | Rate limits recovered |
|---|---|---|---|
| Weak (`TICKET CLASSIFIER. Classify the ticket.`) | 0 of 146 | -- | 18 |
| Strong zero-shot (role + task + the 4 labels + "only the label" + no new labels) | 146 of 146 | 123 of 146 (**84.25%**) | 18 |
| Few-shot (zero-shot + the 13 `few_shot_examples.csv` examples, `Ticket: ...\nLabel: ...` format) | 146 of 146 | 142 of 146 (**97.26%**) | 18 |

`outputs/ticket_classification.csv` holds `ticket_id, true_category, zero_shot, few_shot` for all 146 tickets.

The four tickets few-shot still gets wrong (T0041, T0094, T0110, T0136) all **mention two intents at once** -- e.g. a billing phrase ("recharge went through fine") alongside a plan-change request, or "cancel" alongside a signal complaint, where the *secondary* mentioned intent wins instead of the subscriber's actual (primary) request. Fix: ask the model to identify the subscriber's *primary, action-driving* intent explicitly (e.g. "which intent requires the agent to act first?"), or add a few-shot example that itself contains two intents so the model learns to prioritise one.

Chain-of-thought prompting is worth its extra token cost when the task needs multi-step reasoning the model could get wrong by pattern-matching alone (e.g. multi-constraint arithmetic, policy edge cases). It is not needed here because ticket classification is a short, single-step keyword/semantic match into one of four labels -- the strong/few-shot prompts already reach 84-97% accuracy without any reasoning step, so the extra tokens would not buy a meaningfully better label.

### C. Validated structured extraction with retry and graceful failure (`src/ticket_extraction.py`)

`TicketRecord` (Pydantic): `subscriber_id: Optional[str]`, `category: Literal["billing","signal","plan_change","cancellation","other"]`, `urgency: int = Field(ge=1, le=5)`, `wants_refund: bool`.

| Hand-made input | Result |
|---|---|
| `urgency="3"` (string) | coerced to the integer `3` |
| `urgency=9` | `ValidationError`: field `('urgency',)`, type `less_than_equal` |
| `category="refund"` | `ValidationError`: field `('category',)`, type `literal_error` |

Chain: `PydanticOutputParser(pydantic_object=TicketRecord)`; a system message starting with `TICKET EXTRACTOR` with `parser.get_format_instructions()` inserted via the prompt's `.partial()` method, and a human message that is exactly the ticket text (no prefix/suffix). On a parser failure, the bad reply is appended as an assistant message and a `VALIDATION ERROR: ...` user message is appended, up to **2 retries**; a ticket that still fails is recorded as `manual_review` and the run continues (never crashes). Processed all 150 tickets of `tickets.csv` in file order after `mock_llm.reset()`:

| Outcome | Count |
|---|---|
| Valid on first attempt | **118** |
| Valid after one retry | **28** |
| `manual_review` | **4** (the four unreadable `...` tickets) |
| First-attempt failures (total) | **32** |
| &nbsp;&nbsp;-- not valid JSON | 16 (12 chatty prose + 4 unreadable) |
| &nbsp;&nbsp;-- valid JSON but breaks the schema | 16 (6 with `urgency=7`, 10 with an out-of-list `category`) |
| Rate-limited responses recovered | **23** |

Among the 146 valid records: **31** have `wants_refund = true`, **75** have a null `subscriber_id`. `outputs/ticket_extractions.jsonl` holds one line per ticket (`ticket_id, status, attempts, record`). In production, `manual_review` tickets would be routed to a human agent's queue instead of being auto-processed -- the pipeline never silently guesses a field it couldn't validate.

### D. Retrieval-augmented policy assistant (`src/policy_rag.py`)

**(a) Chunking.** `chunk_text(text, size=400, overlap=60)`: starts at `0, 340, 680, ...` (every `size - overlap` chars), each chunk is `text[start:start+size]`, and a start position is used only while `start + overlap < len(text)`. Applied to the 5 files in `data/policies/`:

| File | Chunks |
|---|---|
| cancellation_policy.txt | 2 |
| plan_change_policy.txt | 3 |
| recharge_and_refund_policy.txt | 3 |
| retention_offers_policy.txt | 3 |
| technical_support_policy.txt | 2 |
| **Total** | **13** |

**(b) Embed and store.** In-memory `chromadb.Client()`, a collection with cosine distance set explicitly (`metadata={"hnsw:space": "cosine"}`), embedded with `embedding_functions.ONNXMiniLM_L6_V2()`, unique chunk ids and `{source, chunk}` metadata. `retrieve(query, k=3)` returns `source`, `similarity = 1 - distance`, `text`.

**(c) Retrieval evaluation** (`data/raw/policy_eval.csv`, 12 answerable + 4 unanswerable questions):

- Top-3 file hits: **12 of 12**; top-1 file hits: **12 of 12**.
- Answerable top-1 similarity range: **0.334 to 0.637** (mean **0.514**).
- Unanswerable top-1 similarities: **0.069, -0.047, 0.295, 0.304**.
- Keyword baseline (shared non-stopword words, ties to the alphabetically first file) top-1 accuracy: **9 of 12**.

Meaning-based search beats keyword matching here because several questions paraphrase the policy wording (e.g. "top-up" / "recharge", "picture breaking up" / "pixelation") that share few or no literal words with the policy text, while the embedding model still places them close in meaning; a keyword overlap score can't see that connection. Of the 4 unanswerable questions, "Is international roaming available on the mobile app?" comes closest to answerable (similarity 0.304) because "plan"/"app"-flavoured policy language overlaps loosely with it, even though no policy actually covers roaming.

**(d) Calibrate and ground.** Rejection threshold = midpoint of the lowest answerable (0.334) and highest unanswerable (0.304) top-1 similarities = **~0.319** (reference ~0.32). The margin between the two ranges is only **~0.03** wide, so a different embedding build could move a borderline question across the line; if a result ever flips, it should be reported, not hidden. `answer(question)` returns the exact fallback sentence `I don't have enough information to answer that.` without calling the model when the best similarity is below the threshold; otherwise it sends a `POLICY ASSISTANT` system message (instructing the model to use only the context, and to say so otherwise) and a user message of the form `CONTEXT:\n[<file>]\n<chunk>\n\n...\n\nQUESTION:\n<question>`. Run over all 16 questions, grounded vs. ungrounded (no threshold, system message `POLICY ASSISTANT. Answer the question.`):

- Grounded: **12 of 12** answerable answers contain the expected fact; **4 of 4** unanswerable questions get the exact fallback sentence.
- Ungrounded: every one of the 16 questions gets the **same confident invented answer** (ignoring the actual question and policy text entirely).

`outputs/rag_eval.csv` holds `question, expected_file, top1_source, top1_similarity, hit_at_3, grounded_answer, ungrounded_answer`.

**(e) Chunk size matters.** Rebuilt with `size=200, overlap=30`: **25 chunks** (vs. 13 at 400/60). For the 12 answerable questions: the expected fact is in the top-3 chunks for **11 of 12** (400/60: 12 of 12); grounded answers contain it for **8 of 12** (400/60: 12 of 12). Smaller chunks lost information because a 200-character window more often splits the one sentence that states the policy fact in half across two chunks, so neither chunk alone contains the whole fact, and a chunk that captures only part of a sentence also scores worse against the question during retrieval.

### E. The Retention Copilot (`src/copilot.py`)

`build_brief(subscriber_id)` runs for the 15 subscribers of `outputs/retention_priority.csv`. For each: reads `tenure_months` and the number of failed recharges (`n_failed_6m`, with SQL `NULL` sums treated as 0 via `COALESCE`) from `data/tataplay.db` with SQL; builds the full feature row with the same feature-building logic as Part B (shared in `src/churn_features.py`, imported by both `churn_model.py` and `copilot.py`; `n_success_6m`/`n_failed_6m`/`recency_days` are computed from recharges as of 2026-03-31, and values read back from SQLite as `None` are converted to `NaN` before scoring). The subscriber is re-scored with `outputs/churn_model.joblib` and the result is asserted to agree with the saved `churn_probability` to 4 decimals; `risk_band` is High (>= 0.40), Medium (>= 0.20) or Low otherwise; the segment name comes from `outputs/segments.csv`; the top-3 policy chunks are retrieved with the query `retention offer for a {risk_band} risk subscriber with {tenure_months} months tenure and {n_failed} failed recharges`. The model is called with a `RETENTION BRIEF` system message (JSON-only, policy-context-only) and a `FACTS:` / `POLICY CONTEXT:` user message (`tenure_months` and `failed_recharges_6m` always sent as plain integers). The reply is validated with Pydantic (`RetentionBrief`: `subscriber_id`, `risk_band` a `Literal["High","Medium","Low"]`, `recommended_action`, `policy_source`), retried up to 2 times the same way as task C, plus consistency checks (`subscriber_id`/`risk_band` must equal what was sent, `policy_source` must be a real policy filename).

- **Valid briefs: 15 of 15.** Per `risk_band`: **14 High, 1 Medium**. Per `recommended_action`: 6 free add-on pack, 4 auto-renew setup call, 4 loyalty discount, 1 "No retention offer applies".
- One subscriber, **TP01015**, has a churn probability of **0.4021** -- right on the High cut-off -- so it is classified High on this run (the reference brief, built on a different library/model build, classified it Medium, giving 13 High / 2 Medium and one more "No offer"). What's graded is that each run's own counts agree with its own probabilities and the rule below; every brief here does.

| Rule (checked against the retrieved policy text) | Action |
|---|---|
| High risk **and** >= 3 failed recharges in 6 months | Auto-renew setup call + 7-day grace extension |
| Else, High risk **and** tenure >= 12 months | Loyalty discount of 15% |
| Else, High risk | Free add-on pack for 2 months |
| Medium or Low risk | No retention offer applies |

Every brief's `policy_source` is `retention_offers_policy.txt`. `outputs/retention_briefs.jsonl` holds one line per subscriber: `subscriber_id, status, attempts, facts, brief`.

## Limitations and Next Steps

- **Synthetic data.** All subscribers, recharges, tickets and churn labels are generated by `generate_data.py`, not real Tata Play data -- the exact numbers (accuracy, F1, costs, similarities) reflect this synthetic generator's statistical structure, not real subscriber behaviour.
- **A single time window.** The churn label, recharge features and revenue-at-risk model are all anchored to one snapshot (2026-03-31) and one six-cycle history; the model has not been validated against a different season or a longer multi-year history, so seasonal effects are unmeasured.
- **A deterministic mock instead of a real model.** `mock_llm.py` is a rule-based stand-in -- ticket classification, extraction, retrieval grounding and retention briefs all depend on its fixed keyword/regex behaviour, which will not perfectly predict how a real hosted LLM responds to the same prompts (a real model could be more robust, or fail in different ways).
- **The revenue-at-risk/retention-brief lists are built from test-set subscribers whose outcome is already known.** They illustrate the method on labelled data; they are not a forward-looking evaluation on genuinely unseen subscribers, and churn probabilities for new subscribers in production have not been separately validated.
- **Thin retrieval-threshold margin.** The rejection threshold (~0.32) sits in a window only ~0.03 wide between the lowest answerable and highest unanswerable similarity -- a different embedding model/version, or a slightly different real-world question, could easily cross that line and flip an answerable question to "I don't have enough information" or vice versa.
- **Small, templated policy corpus.** Only 5 short policy files exist, each covering one topic; a production system would need many more documents, multi-topic chunks, and a higher-recall retrieval evaluation set before the thresholds here could be trusted at scale.

## Running the pipeline yourself

```powershell
python generate_data.py          # 1. generate raw data
python src/audit_data.py         # 2. inspect raw data quality
python src/clean_subscribers.py  # 3. clean subscribers (must run before recharges)
python src/clean_recharges.py    # 4. clean recharges (depends on cleaned subscribers)
python src/clean_plans.py        # 5. copy plans unchanged
python src/store_db.py           # 6. write the three clean tables to data/tataplay.db
python src/run_queries.py        # 7. run sql/queries.sql and print the results
python src/churn_model.py        # 8. Part B: train, tune, select and persist the churn model
python src/segments.py           # 9. Part C: K-means segmentation -> outputs/segments.csv
python src/spend_model.py        # 10. Part C: collinearity check + Ridge/Lasso spend model
python src/revenue_at_risk.py    # 11. Part C: revenue-at-risk priority list -> outputs/retention_priority.csv
python src/llm_client_demo.py    # 12. Part D: call_llm() demonstrations (401/404/length/memory/rate limit)
python src/ticket_classifier.py  # 13. Part D: weak/zero-shot/few-shot ticket classification
python src/ticket_extraction.py  # 14. Part D: validated TicketRecord extraction with retry
python src/policy_rag.py         # 15. Part D: chunking + retrieval evaluation + grounded answers
python src/copilot.py            # 16. Part D: Retention Copilot -> outputs/retention_briefs.jsonl
```

To point `llm_client.py` at a real endpoint instead of the mock, set `LLM_MODE=live` and `OPENROUTER_API_KEY=<your key>` in the environment before running any Part D script -- no code change is needed, and no key is ever stored in this repository.

