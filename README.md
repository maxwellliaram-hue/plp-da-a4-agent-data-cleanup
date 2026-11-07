# Mobile Money Agent Network: Data Clean-Up

Cleans two messy extracts from a mobile-money agent network (transactions and agents) into one analysis-ready table. Every cleaning rule lives in code (`cleaning_pipeline.py`), is logged with its data-quality dimension and row count, and the notebook produces a county summary, a weekly pivot and a one-page Data Quality Report.

## Before and after

| Stage | Transactions | Agents |
|---|---|---|
| Raw (`generate_data.py`, seed 44) | **2,075** | 63 |
| Clean (`outputs/agent_transactions_clean.csv`) | **1,948** | 60 unique |

127 transaction rows removed: 60 exact duplicates, 15 conflicting duplicate `txn_id`s, 12 missing `txn_type`, 20 missing `txn_date`, 20 dates outside 2026-01-01 to 2026-06-30. A further 56 amounts were imputed and flagged, 114 outliers flagged, and 38 transactions have agents missing from the agents table (kept, with `county = "Unknown"`).

## Repository contents

| File | Purpose |
|---|---|
| `generate_data.py` | Creates `raw/transactions_raw.csv` and `raw/agents_raw.csv` (seed 44) |
| `cleaning_pipeline.py` | `clean_agent_data(txns, agents)` returns `(clean_df, log)` |
| `agent_cleanup.ipynb` | Audit table, post-merge validation, county summary, weekly pivot, Data Quality Report |
| `outputs/agent_transactions_clean.csv` | Final clean dataset |
| `requirements.txt` | `pandas`, `numpy` |

## How to run

```bash
pip install -r requirements.txt
pip install jupyter            # only needed to open/run the notebook

python generate_data.py        # regenerates raw/*.csv (git-ignored, not committed)
jupyter nbconvert --to notebook --execute --inplace agent_cleanup.ipynb
```

Or open `agent_cleanup.ipynb`, choose *Restart Kernel and Run All*. The notebook regenerates the raw data itself, calls `clean_agent_data()` and rewrites `outputs/agent_transactions_clean.csv`.

To use the pipeline directly:

```python
import pandas as pd
from cleaning_pipeline import clean_agent_data

txns = pd.read_csv("raw/transactions_raw.csv", dtype=str)
agents = pd.read_csv("raw/agents_raw.csv", dtype=str)
clean_df, log = clean_agent_data(txns, agents)
```

## Cleaning rules (by data-quality dimension)

- **Completeness:** drop rows with no `txn_id`, `txn_type` or `txn_date`; impute missing amounts with the median for the same `txn_type` and flag `amount_imputed`; missing county becomes `Unknown`.
- **Validity:** non-numeric and zero/negative amounts are imputed and flagged; unrecognised types and unparseable dates are dropped.
- **Uniqueness:** exact duplicates and repeated `txn_id`s removed (first record kept); duplicate agent rows removed so the merge is many-to-one.
- **Consistency:** `txn_type` limited to Deposit/Withdrawal, dates to ISO `YYYY-MM-DD`, amounts to numeric, `agent_id` upper-cased and trimmed, county title-cased.
- **Accuracy:** amounts outside 1.5 x IQR within `txn_type` are flagged `is_outlier` (kept for review).
- **Timeliness:** transactions outside the reporting window (parameters `period_start`, `period_end`) are removed.

## Output columns

`txn_id, agent_id, agent_name, county, tier, txn_type, amount, txn_date, amount_imputed, is_outlier, agent_known`
