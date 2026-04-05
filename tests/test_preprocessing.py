"""Tests for src.data.preprocessing."""

import pandas as pd
from src.data.preprocessing import CensusPipeline, encode_features


class TestEncodeFeatures:
    def test_output_types(self, features_and_income):
        features_raw, income_raw = features_and_income
        features_final, income = encode_features(features_raw, income_raw)
        assert isinstance(features_final, pd.DataFrame)
        assert isinstance(income, pd.Series)

    def test_income_binary(self, features_and_income):
        _, income_raw = features_and_income
        _, income = encode_features(pd.DataFrame({"a": [1, 2]}), income_raw)
        assert set(income.unique()).issubset({0, 1})


class TestCensusPipelineNoBinning:
    def test_output_all_numeric(self, features_and_income):
        features_raw, income_raw = features_and_income
        pipeline = CensusPipeline(use_binning=False)
        features_final, _ = pipeline.fit_transform(features_raw, income_raw)
        for c in features_final.columns:
            assert pd.api.types.is_numeric_dtype(features_final[c]) or pd.api.types.is_bool_dtype(
                features_final[c]
            ), f"Column {c} has dtype {features_final[c].dtype}"

    def test_income_binary(self, features_and_income):
        features_raw, income_raw = features_and_income
        pipeline = CensusPipeline(use_binning=False)
        _, income = pipeline.fit_transform(features_raw, income_raw)
        assert set(income.unique()).issubset({0, 1})

    def test_has_interaction_features(self, features_and_income):
        features_raw, income_raw = features_and_income
        pipeline = CensusPipeline(use_binning=False)
        features_final, _ = pipeline.fit_transform(features_raw, income_raw)
        assert "age_x_hours" in features_final.columns
        assert "edu_x_capgain" in features_final.columns

    def test_no_binned_features(self, features_and_income):
        features_raw, income_raw = features_and_income
        pipeline = CensusPipeline(use_binning=False)
        features_final, _ = pipeline.fit_transform(features_raw, income_raw)
        assert "hours_bin" not in features_final.columns

    def test_transform_aligns_columns(self, features_and_income):
        features_raw, income_raw = features_and_income
        pipeline = CensusPipeline(use_binning=False)
        features_final, _ = pipeline.fit_transform(features_raw, income_raw)
        transformed = pipeline.transform(features_raw.head(10))
        assert list(transformed.columns) == list(features_final.columns)


class TestCensusPipelineWithBinning:
    def test_has_binned_features(self, features_and_income):
        features_raw, income_raw = features_and_income
        pipeline = CensusPipeline(use_binning=True)
        features_final, _ = pipeline.fit_transform(features_raw, income_raw)
        assert "hours_bin" in features_final.columns

    def test_more_features_than_no_binning(self, features_and_income):
        features_raw, income_raw = features_and_income
        no_bin = CensusPipeline(use_binning=False)
        with_bin = CensusPipeline(use_binning=True)
        f_no, _ = no_bin.fit_transform(features_raw, income_raw)
        f_yes, _ = with_bin.fit_transform(features_raw, income_raw)
        assert f_yes.shape[1] > f_no.shape[1]

    def test_same_income_encoding(self, features_and_income):
        features_raw, income_raw = features_and_income
        _, inc_no = CensusPipeline(use_binning=False).fit_transform(features_raw, income_raw)
        _, inc_yes = CensusPipeline(use_binning=True).fit_transform(features_raw, income_raw)
        pd.testing.assert_series_equal(
            inc_no.reset_index(drop=True), inc_yes.reset_index(drop=True)
        )
