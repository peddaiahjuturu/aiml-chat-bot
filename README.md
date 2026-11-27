# AIML Chat Bot -- Tata Play Subscriber Retention & Support Copilot (Capstone)

Synthetic-data pipeline and offline mock LLM for a subscriber retention / support chatbot capstone project.

## Project structure

```
generate_data.py        # creates all synthetic raw data + policy docs (stdlib only)
mock_llm.py              # offline, deterministic stand-in for a hosted chat-completion API
src/
  audit_data.py          # reports shape/dtypes/missing values for the raw tables
  clean_subscribers.py   # cleans data/raw/subscribers.csv -> data/clean/subscribers.csv
  clean_recharges.py     # cleans data/raw/recharges.csv  -> data/clean/recharges.csv
data/
  raw/                   # generated, uncleaned source tables
  clean/                 # cleaned tables produced by the src/clean_*.py scripts
  policies/              # policy text documents used by the retention/support copilot
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
2. Audited the raw tables with `src/audit_data.py` to understand shape, dtypes, and missing values before cleaning.
3. Cleaned `subscribers.csv` and `recharges.csv` with `src/clean_subscribers.py` and `src/clean_recharges.py`, writing the results to `data/clean/`.

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

### Cleaning results

**`python src/clean_subscribers.py`** (`data/raw/subscribers.csv` -> `data/clean/subscribers.csv`)

- Output rows: **2,400** (started from 2,415)
- Cities standardised to 7 canonical names: `Bengaluru, Chennai, Delhi, Hyderabad, Kolkata, Mumbai, Pune`
- `auto_renew` standardised to: `No, Yes`
- Rules applied:
  - Removed exact duplicate rows: 15
  - Standardised city to one of 7 canonical names: 892
  - Standardised `auto_renew` to Yes/No: 695
  - Set `age` to missing where outside the plausible 10-100 range: 6

**`python src/clean_recharges.py`** (`data/raw/recharges.csv` -> `data/clean/recharges.csv`)

- Output rows: **12,985** (started from 13,055)
- `status` values standardised to: `failed, pending, success`
- Rules applied:
  - Removed exact duplicate rows: 60
  - Removed recharges for unknown subscribers (not present in cleaned subscribers table): 10
  - Lower-cased `status`: 5,099
  - Rebuilt `amount_inr` as a number, stripping a leading `"INR "` text prefix: 364
  - Parsed `recharge_date` to a date type (unparsable values set aside): 0

No missing value is ever filled in by the cleaning scripts -- cleaning only removes rows and standardises existing values.

## Running the pipeline yourself

```powershell
python generate_data.py          # 1. generate raw data
python src/audit_data.py         # 2. inspect raw data quality
python src/clean_subscribers.py  # 3. clean subscribers (must run before recharges)
python src/clean_recharges.py    # 4. clean recharges (depends on cleaned subscribers)
```
