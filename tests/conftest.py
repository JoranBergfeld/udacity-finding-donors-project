"""Shared test fixtures."""

import pandas as pd
import pytest
from src.data.loader import load_census


@pytest.fixture(scope="session")
def census_df() -> pd.DataFrame:
    """Full census dataset."""
    return load_census("starter/census.csv")


@pytest.fixture(scope="session")
def sample_census_df(census_df) -> pd.DataFrame:
    """First 100 rows for fast tests."""
    return census_df.head(100).copy()


@pytest.fixture(scope="session")
def features_and_income(sample_census_df):
    """Pre-split features and income from sample data."""
    income_raw = sample_census_df["income"]
    features_raw = sample_census_df.drop("income", axis=1)
    return features_raw, income_raw
