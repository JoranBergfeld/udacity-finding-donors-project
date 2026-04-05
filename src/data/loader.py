"""Data loading and exploration utilities for the census dataset."""

from pathlib import Path

import pandas as pd


def load_census(path: str | Path = "starter/census.csv") -> pd.DataFrame:
    """Load census CSV, stripping whitespace from categorical columns."""
    data = pd.read_csv(path)
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].str.strip()
    return data


def load_test(path: str | Path = "starter/test_census.csv") -> pd.DataFrame:
    """Load Kaggle test CSV (has unnamed index column, no income, possible NaNs)."""
    data = pd.read_csv(path, index_col=0)
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].str.strip()
    return data


def explore_data(data: pd.DataFrame) -> dict:
    """Return summary statistics matching notebook Cell 6 expectations.

    Returns dict with keys: n_records, n_greater_50k, n_at_most_50k, greater_percent.
    Handles both raw string income ('<=50K', '>50K') and pre-encoded numeric (0, 1).
    """
    n_records = len(data)

    income = data["income"]
    if pd.api.types.is_numeric_dtype(income):
        n_greater_50k = int(income.sum())
    else:
        n_greater_50k = int((income == ">50K").sum())

    n_at_most_50k = n_records - n_greater_50k
    greater_percent = (n_greater_50k / n_records) * 100

    return {
        "n_records": n_records,
        "n_greater_50k": n_greater_50k,
        "n_at_most_50k": n_at_most_50k,
        "greater_percent": greater_percent,
    }
