"""Cleaning pipeline for the Mobile Money Agent Network data.

Every rule is applied in code. There are no hard-coded row numbers and no
manual edits; each rule is logged with its data-quality dimension and the
number of rows it affected.
"""
import numpy as np
import pandas as pd

TXN_TYPE_MAP = {
    "deposit": "Deposit", "dep": "Deposit", "cash-in": "Deposit", "cash in": "Deposit",
    "withdrawal": "Withdrawal", "withdraw": "Withdrawal", "w/d": "Withdrawal",
    "cash out": "Withdrawal", "cash-out": "Withdrawal",
}
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"]
OUTPUT_COLUMNS = ["txn_id", "agent_id", "agent_name", "county", "tier", "txn_type",
                  "amount", "txn_date", "amount_imputed", "is_outlier", "agent_known"]


def _strip_blank(s):
    """Trim whitespace and turn empty strings into missing values."""
    s = s.astype("string").str.strip()
    return s.mask(s == "", pd.NA)


def _parse_dates(s):
    """Parse text dates written in any of DATE_FORMATS. Returns (parsed, is_iso)."""
    parsed = None
    for fmt in DATE_FORMATS:
        p = pd.to_datetime(s, format=fmt, errors="coerce")
        parsed = p if parsed is None else parsed.fillna(p)
    is_iso = pd.to_datetime(s, format=DATE_FORMATS[0], errors="coerce").notna()
    return parsed, is_iso


def clean_agent_data(txns, agents, period_start="2026-01-01", period_end="2026-06-30"):
    """Clean the raw transactions and agents extracts and merge them.

    Parameters
    ----------
    txns : DataFrame
        Raw transactions (txn_id, agent_id, txn_type, amount, txn_date).
    agents : DataFrame
        Raw agents (agent_id, agent_name, county, tier, ...).
    period_start, period_end : str
        Reporting window (ISO dates). Transactions outside it fail the
        timeliness check and are removed.

    Returns
    -------
    clean_df : DataFrame
        One row per unique txn_id with standardised txn_type, numeric amount,
        ISO txn_date (YYYY-MM-DD), the agent's county/name/tier from the merge,
        and the flags amount_imputed, is_outlier and agent_known.
    log : list of dict
        One entry per rule with keys: issue, dimension, column, rows_affected.
    """
    log = []

    def add(issue, dimension, column, n):
        log.append({"issue": issue, "dimension": dimension,
                    "column": column, "rows_affected": int(n)})

    # ------------------------------------------------------------ agents table
    a = agents.copy()
    for c in a.columns:
        a[c] = _strip_blank(a[c])
    a["agent_id"] = a["agent_id"].str.upper()
    county_raw = a["county"].copy()
    a["county"] = a["county"].str.title()
    add("Agents: county text inconsistent (case / padding)", "Consistency",
        "county", (county_raw.notna() & (county_raw != a["county"])).sum())
    add("Agents: county missing, set to 'Unknown'", "Completeness",
        "county", a["county"].isna().sum())
    a["county"] = a["county"].fillna("Unknown")
    a["tier"] = a["tier"].str.title().fillna("Unknown")
    a["agent_name"] = a["agent_name"].fillna("Unknown")
    n_dup_agents = a.duplicated("agent_id").sum()
    add("Agents: duplicate agent rows removed", "Uniqueness", "agent_id", n_dup_agents)
    a = a.drop_duplicates("agent_id", keep="first")

    # -------------------------------------------------- transactions: text tidy
    t = txns.copy()
    for c in t.columns:
        t[c] = _strip_blank(t[c])
    n_raw = len(t)

    add("txn_id missing, row removed", "Completeness", "txn_id", t["txn_id"].isna().sum())
    t = t[t["txn_id"].notna()]

    agent_before = t["agent_id"].copy()
    t["agent_id"] = t["agent_id"].str.upper()
    add("agent_id typed in lower case / with padding", "Consistency", "agent_id",
        (agent_before.notna() & (agent_before != t["agent_id"])).sum())

    type_before = t["txn_type"].copy()
    t["txn_type"] = type_before.str.lower().map(TXN_TYPE_MAP)
    add("txn_type spelled inconsistently (case, padding, aliases)", "Consistency", "txn_type",
        (type_before.notna() & t["txn_type"].notna() & (type_before != t["txn_type"])).sum())

    # -------------------------------------------------------------- uniqueness
    exact = t.duplicated(keep="first")
    add("Exact duplicate transactions removed", "Uniqueness", "txn_id", exact.sum())
    t = t[~exact]
    conflict = t.duplicated("txn_id", keep="first")
    add("Repeated txn_id with conflicting values (first record kept)", "Uniqueness",
        "txn_id", conflict.sum())
    t = t[~conflict]

    # ---------------------------------------------------------------- txn_type
    unmapped = type_before.reindex(t.index).notna() & t["txn_type"].isna()
    add("txn_type missing, row removed", "Completeness", "txn_type",
        type_before.reindex(t.index).isna().sum())
    add("txn_type not recognised, row removed", "Validity", "txn_type", unmapped.sum())
    t = t[t["txn_type"].notna()]

    # ---------------------------------------------------------------- txn_date
    parsed, is_iso = _parse_dates(t["txn_date"])
    add("txn_date missing, row removed", "Completeness", "txn_date", t["txn_date"].isna().sum())
    add("txn_date unparseable, row removed", "Validity", "txn_date",
        (t["txn_date"].notna() & parsed.isna()).sum())
    add("txn_date not in ISO format (dd/mm/yyyy or 'Mon dd, yyyy')", "Consistency",
        "txn_date", (parsed.notna() & ~is_iso).sum())
    in_window = parsed.between(pd.Timestamp(period_start), pd.Timestamp(period_end))
    add("txn_date outside the reporting window, row removed", "Timeliness", "txn_date",
        (parsed.notna() & ~in_window).sum())
    t = t.assign(txn_date=parsed)[in_window]

    # ------------------------------------------------------------------ amount
    raw_amt = t["amount"]
    text = (raw_amt.str.replace(r"(?i)\b(kes|ksh)\b", "", regex=True)
                   .str.replace(",", "", regex=False).str.strip())
    num = pd.to_numeric(text, errors="coerce")
    add("amount has currency prefix / thousands separators", "Consistency", "amount",
        (raw_amt.notna() & num.notna() & (raw_amt != text)).sum())
    add("amount missing, imputed", "Completeness", "amount", raw_amt.isna().sum())
    add("amount not numeric (text), imputed", "Validity", "amount",
        (raw_amt.notna() & num.isna()).sum())
    add("amount zero or negative, imputed", "Validity", "amount", (num <= 0).sum())
    num = num.mask(num <= 0)
    t["amount"] = num.astype("float64")
    t["amount_imputed"] = t["amount"].isna()
    t["amount"] = t["amount"].fillna(t.groupby("txn_type")["amount"].transform("median"))

    # ---------------------------------------------------------------- outliers
    genuine = t["amount"].where(~t["amount_imputed"])
    q1 = genuine.groupby(t["txn_type"]).transform(lambda s: s.quantile(0.25))
    q3 = genuine.groupby(t["txn_type"]).transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    t["is_outlier"] = (t["amount"] > q3 + 1.5 * iqr) | (t["amount"] < q1 - 1.5 * iqr)
    add("Extreme amounts flagged (1.5 x IQR within txn_type), kept for review",
        "Accuracy", "amount", t["is_outlier"].sum())

    # ------------------------------------------------------------------- merge
    merged = t.merge(a[["agent_id", "agent_name", "county", "tier"]],
                     on="agent_id", how="left", validate="many_to_one", indicator=True)
    merged["agent_known"] = merged["_merge"] == "both"
    merged = merged.drop(columns="_merge")
    add("agent_id not found in agents table (county set to 'Unknown')", "Consistency",
        "agent_id", (~merged["agent_known"]).sum())
    for c in ["agent_name", "county", "tier"]:
        merged[c] = merged[c].fillna("Unknown")

    clean = (merged.assign(txn_date=merged["txn_date"].dt.strftime("%Y-%m-%d"))
                   [OUTPUT_COLUMNS]
                   .sort_values(["txn_date", "txn_id"]).reset_index(drop=True))
    for c in ["txn_id", "agent_id", "txn_type", "agent_name", "county", "tier"]:
        clean[c] = clean[c].astype(object)
    return clean, log
