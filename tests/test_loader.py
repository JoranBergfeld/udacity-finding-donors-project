"""Tests for src.data.loader."""

from src.data.loader import explore_data, load_census, load_test


def test_load_census_shape():
    data = load_census("starter/census.csv")
    assert data.shape[1] == 14
    assert len(data) > 0


def test_load_census_stripped_whitespace():
    data = load_census("starter/census.csv")
    for col in data.select_dtypes(include=["object", "string"]).columns:
        assert not any(data[col].str.startswith(" "))


def test_load_test_shape():
    data = load_test("starter/test_census.csv")
    assert data.shape[1] == 13  # No income column
    assert len(data) > 0


def test_load_test_no_unnamed_index():
    data = load_test("starter/test_census.csv")
    assert "Unnamed: 0" not in data.columns


def test_explore_data_keys(census_df):
    stats = explore_data(census_df)
    assert set(stats.keys()) == {"n_records", "n_greater_50k", "n_at_most_50k", "greater_percent"}


def test_explore_data_values_sum(census_df):
    stats = explore_data(census_df)
    assert stats["n_records"] == stats["n_greater_50k"] + stats["n_at_most_50k"]


def test_explore_data_percent_range(census_df):
    stats = explore_data(census_df)
    assert 0 < stats["greater_percent"] < 100
