"""Dataset generator for the Mobile Money Agent Network data clean-up (seed 44).

Creates two deliberately messy raw extracts in ./raw/:
    agents_raw.csv        - one row per agent (with county, tier, ...)
    transactions_raw.csv  - one row per transaction (deposits / withdrawals)

Injected problems (all reproducible because the seed is fixed):
    * duplicate transactions (exact copies and conflicting copies of a txn_id)
    * inconsistent txn_type spellings / case / whitespace
    * amounts stored as text ("1,250.00", "KES 900", "abc", "N/A", negatives, blanks)
    * three different date formats, blank dates, future and very old dates
    * agent_id typos in case / whitespace and orphan agent_ids not in the agent list
    * extreme high-value amounts
    * messy county names, missing counties and duplicate agent rows

Run:  python generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 44
N_AGENTS = 60
N_TXNS = 2000
OUT_DIR = Path("raw")

COUNTIES = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Uasin Gishu",
            "Kiambu", "Machakos", "Kakamega"]
COUNTY_P = [0.28, 0.14, 0.10, 0.10, 0.08, 0.12, 0.09, 0.09]


def build_datasets(seed=SEED):
    rng = np.random.default_rng(seed)

    def pick(n, k):
        return rng.choice(n, size=k, replace=False)

    # ------------------------------------------------------------------ agents
    agent_ids = [f"AG{i:03d}" for i in range(1, N_AGENTS + 1)]
    agents = pd.DataFrame({
        "agent_id": agent_ids,
        "agent_name": [f"Agent {i:03d}" for i in range(1, N_AGENTS + 1)],
        "county": rng.choice(COUNTIES, size=N_AGENTS, p=COUNTY_P),
        "tier": rng.choice(["Bronze", "Silver", "Gold"], size=N_AGENTS, p=[0.5, 0.35, 0.15]),
        "onboarded": (pd.Timestamp("2022-01-01")
                      + pd.to_timedelta(rng.integers(0, 1400, N_AGENTS), unit="D")
                      ).strftime("%Y-%m-%d"),
    }).astype(object)

    for i in pick(N_AGENTS, 10):                    # inconsistent county text
        style = rng.choice(["upper", "lower", "pad"])
        c = agents.loc[i, "county"]
        agents.loc[i, "county"] = {"upper": c.upper(), "lower": c.lower(),
                                   "pad": f"  {c} "}[style]
    agents.loc[pick(N_AGENTS, 4), "county"] = None  # missing county
    agents = pd.concat([agents, agents.sample(3, random_state=seed)],  # duplicate agent rows
                       ignore_index=True)

    # ------------------------------------------------------------ transactions
    n = N_TXNS
    txn_id = [f"TX{i:05d}" for i in range(1, n + 1)]

    # agent_id: valid ids, some orphans, some case/space typos
    agent_col = rng.choice(agent_ids, size=n).astype(object)
    orphan_idx = pick(n, 40)
    agent_col[orphan_idx] = rng.choice(["AG091", "AG092", "AG093", "AG094", "AG095"],
                                       size=40)
    typo_idx = pick(n, 30)
    agent_col[typo_idx] = [f" {a.lower()} " if k % 2 else a.lower()
                           for k, a in enumerate(agent_col[typo_idx])]

    # txn_type: canonical + messy variants
    txn_type = rng.choice(["Deposit", "Withdrawal"], size=n, p=[0.55, 0.45]).astype(object)
    variants = {
        "Deposit": ["deposit", "DEPOSIT", " Deposit ", "Dep", "cash-in"],
        "Withdrawal": ["withdrawal", "WITHDRAWAL", " Withdrawal ", "Withdraw", "w/d", "cash out"],
    }
    for i in pick(n, 500):
        txn_type[i] = rng.choice(variants[txn_type[i]])
    txn_type[pick(n, 12)] = None                    # missing type

    # amount: lognormal, extreme outliers, then text formatting noise
    amount = np.round(rng.lognormal(np.log(2500), 0.7, n), 2)
    out_idx = pick(n, 15)
    amount[out_idx] = np.round(amount[out_idx] * rng.uniform(30, 60, 15), 2)
    amount_s = np.array([f"{v:.2f}" for v in amount], dtype=object)
    for i in pick(n, 150):
        amount_s[i] = f"{amount[i]:,.2f}"
    for i in pick(n, 80):
        amount_s[i] = f"KES {amount[i]:,.0f}"
    for i in pick(n, 20):
        amount_s[i] = rng.choice(["abc", "N/A", "?"])
    for i in pick(n, 15):
        amount_s[i] = f"-{amount[i]:.2f}"
    amount_s[pick(n, 25)] = None

    # txn_date: Jan 1 - Jun 30 2026, three formats, plus bad dates
    dates = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 181, n), unit="D")
    fmt_choice = rng.choice([0, 1, 2], size=n, p=[0.70, 0.15, 0.15])
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"]
    date_s = np.array([d.strftime(fmts[f]) for d, f in zip(dates, fmt_choice)], dtype=object)
    for i in pick(n, 12):                           # future dates
        date_s[i] = f"2027-0{rng.integers(1, 9)}-1{rng.integers(0, 9)}"
    for i in pick(n, 8):                            # stale dates
        date_s[i] = f"2019-0{rng.integers(1, 9)}-2{rng.integers(0, 9)}"
    date_s[pick(n, 20)] = None

    txns = pd.DataFrame({
        "txn_id": txn_id,
        "agent_id": agent_col,
        "txn_type": txn_type,
        "amount": amount_s,
        "txn_date": date_s,
    })

    # duplicates: exact copies + conflicting copies (same txn_id, different amount)
    exact = txns.sample(60, random_state=seed)
    conflict = txns.sample(15, random_state=seed + 1).copy()
    conflict["amount"] = [f"{rng.uniform(100, 9000):.2f}" for _ in range(len(conflict))]
    txns = pd.concat([txns, exact, conflict], ignore_index=True)
    txns = txns.sample(frac=1, random_state=seed).reset_index(drop=True)
    return txns, agents


def main():
    txns, agents = build_datasets()
    OUT_DIR.mkdir(exist_ok=True)
    txns.to_csv(OUT_DIR / "transactions_raw.csv", index=False)
    agents.to_csv(OUT_DIR / "agents_raw.csv", index=False)
    print(f"transactions_raw.csv: {len(txns)} rows | agents_raw.csv: {len(agents)} rows")


if __name__ == "__main__":
    main()
