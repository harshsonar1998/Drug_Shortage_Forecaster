"""
drug_shortage_forecaster.data.processor
-----------------------------------------
Converts FDA shortage records into monthly time-series signals.

Since the FDA API provides only 'initial_posting_date' (not start/end),
we build the signal differently:
  - Count how many records were POSTED per month per drug
  - Compute month-over-month log-change of that count
  - This "posting rate volatility" captures how erratically a drug
    enters shortage — high volatility = unpredictable supply behaviour
"""

import numpy as np
import pandas as pd
from typing import Optional

_EPSILON = 1e-6


def list_drugs(df: pd.DataFrame, min_records: int = 3) -> list[str]:
    """Return sorted list of drug names with enough records."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if min_records < 1:
        raise ValueError("min_records must be >= 1")
    if df.empty or "drug_name" not in df.columns:
        return []
    counts = df["drug_name"].value_counts()
    return sorted(counts[counts >= min_records].index.tolist())


def build_shortage_series(
    df: pd.DataFrame,
    drug_name: str,
    freq: str = "ME",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.Series:
    """Build monthly log-change signal for one drug.

    Uses 'initial_posting_date' to count how many shortage records
    were posted per month, then returns the log-change of that count.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    drug_upper = drug_name.strip().upper()
    mask = df["drug_name"].str.upper().str.strip() == drug_upper
    subset = df[mask].copy()

    if subset.empty:
        raise ValueError(
            f"No shortage records found for drug '{drug_name}'. "
            f"Use list_drugs() to see available names."
        )

    # Use initial_posting_date or update_date as the signal date
    date_col = "initial_posting_date" if "initial_posting_date" in subset.columns else "update_date"
    valid_dates = subset[date_col].dropna()

    if valid_dates.empty:
        raise ValueError(f"No valid dates for drug '{drug_name}'.")

    range_start = pd.to_datetime(start) if start else valid_dates.min().replace(day=1)
    range_end   = pd.to_datetime(end)   if end   else pd.Timestamp.today()

    monthly_index = pd.date_range(range_start, range_end, freq=freq)
    if len(monthly_index) < 2:
        raise ValueError(f"Date range too short for '{drug_name}'.")

    # Count postings per month
    subset = subset.copy()
    subset["month"] = subset[date_col].dt.to_period("M")
    monthly_counts = subset.groupby("month").size()

    counts = pd.Series(0.0, index=monthly_index)
    for month in monthly_index:
        period = month.to_period("M")
        counts[month] = float(monthly_counts.get(period, 0))

    # Log-change
    log_counts  = np.log(counts + _EPSILON)
    log_changes = log_counts.diff().dropna()

    if len(log_changes) < 2:
        raise ValueError(
            f"Too few monthly observations for '{drug_name}' (got {len(log_changes)}, need >= 2)."
        )

    log_changes.name = drug_upper
    return log_changes


def build_activity_counts(
    df: pd.DataFrame,
    drug_name: str,
    freq: str = "ME",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.Series:
    """Return raw monthly posting counts for a drug."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    drug_upper = drug_name.strip().upper()
    mask = df["drug_name"].str.upper().str.strip() == drug_upper
    subset = df[mask].copy()

    if subset.empty:
        return pd.Series(dtype=float, name=drug_upper)

    date_col = "initial_posting_date" if "initial_posting_date" in subset.columns else "update_date"
    valid_dates = subset[date_col].dropna()
    if valid_dates.empty:
        return pd.Series(dtype=float, name=drug_upper)

    range_start = pd.to_datetime(start) if start else valid_dates.min().replace(day=1)
    range_end   = pd.to_datetime(end)   if end   else pd.Timestamp.today()

    monthly_index = pd.date_range(range_start, range_end, freq=freq)
    subset["month"] = subset[date_col].dt.to_period("M")
    monthly_counts = subset.groupby("month").size()

    counts = pd.Series(0.0, index=monthly_index, name=drug_upper)
    for month in monthly_index:
        period = month.to_period("M")
        counts[month] = float(monthly_counts.get(period, 0))

    return counts
