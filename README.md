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
sql/
  queries.sql            # the four analysis queries (Q1-Q4)
data/
  raw/                   # generated, uncleaned source tables
  clean/                 # cleaned tables produced by the src/clean_*.py scripts
  policies/              # policy text documents used by the retention/support copilot
  tataplay.db             # SQLite database holding the clean subscribers/recharges/plans tables
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

## Running the pipeline yourself

```powershell
python generate_data.py          # 1. generate raw data
python src/audit_data.py         # 2. inspect raw data quality
python src/clean_subscribers.py  # 3. clean subscribers (must run before recharges)
python src/clean_recharges.py    # 4. clean recharges (depends on cleaned subscribers)
python src/clean_plans.py        # 5. copy plans unchanged
python src/store_db.py           # 6. write the three clean tables to data/tataplay.db
python src/run_queries.py        # 7. run sql/queries.sql and print the results
```
