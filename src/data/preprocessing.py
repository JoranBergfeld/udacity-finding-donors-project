"""
Data preprocessing pipeline for the census dataset.

Single pipeline with a binning toggle:
- CensusPipeline(use_binning=False): log + MinMaxScaler + get_dummies + interaction features
- CensusPipeline(use_binning=True): same + binned features from extensible list
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

BINNED_FEATURE_NAMES = ["hours_bin"]
CONTINUOUS_FEATURES = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
SKEWED_FEATURES = ["capital-gain", "capital-loss"]
CATEGORICAL_FEATURES = [
    "workclass",
    "education_level",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def _encode_income(income_raw: pd.Series) -> pd.Series:
    """Convert income labels to binary."""
    if pd.api.types.is_numeric_dtype(income_raw):
        return income_raw.astype(int)
    return (income_raw.astype(str).str.strip() == ">50K").astype(int)


def encode_features(features_log_minmax_transform: pd.DataFrame, income_raw: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Notebook helper: one-hot encode pre-processed features + encode income.
    Called after the notebook's provided cells have already done log-transform and MinMaxScaler.
    """
    features_final = pd.get_dummies(features_log_minmax_transform)
    income = _encode_income(income_raw)
    return features_final, income


class CensusPipeline:
    """
    Single pipeline with binning toggle. Uses MinMaxScaler to match notebook.
    Fixed steps: log-transform skewed -> MinMaxScaler -> get_dummies -> interaction features.
    Toggle: use_binning appends binned features from an extensible list.
    """

    def __init__(self, use_binning: bool = False):
        self.use_binning = use_binning
        self._scaler = MinMaxScaler() # Use same scaler for fit and transform to ensure consistent scaling
        self._dummy_columns: list[str] | None = None
        self._all_columns: list[str] | None = None

    def fit_transform(self, features_raw: pd.DataFrame, income_raw: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        features = features_raw.copy()

        # Log-transform skewed features
        features[SKEWED_FEATURES] = features[SKEWED_FEATURES].apply(lambda x: np.log(x + 1))

        # MinMaxScaler on continuous features
        features[CONTINUOUS_FEATURES] = self._scaler.fit_transform(features[CONTINUOUS_FEATURES])

        # One-hot encode categoricals
        features = pd.get_dummies(features)
        self._dummy_columns = features.columns.tolist()

        # Always add interaction features
        features = self._add_interaction_features(features_raw, features)

        # Optionally add binned features
        if self.use_binning:
            features = self._add_binned_features(features_raw, features)

        self._all_columns = features.columns.tolist()
        income = _encode_income(income_raw)
        return features, income

    def transform(self, features_raw: pd.DataFrame) -> pd.DataFrame:
        features = features_raw.copy()
        features[SKEWED_FEATURES] = features[SKEWED_FEATURES].apply(lambda x: np.log(x + 1))
        features[CONTINUOUS_FEATURES] = self._scaler.transform(features[CONTINUOUS_FEATURES])
        features = pd.get_dummies(features)
        # Align dummy columns with training set
        features = features.reindex(columns=self._dummy_columns, fill_value=0)
        features = self._add_interaction_features(features_raw, features)
        if self.use_binning:
            features = self._add_binned_features(features_raw, features)
        return features

    def _add_interaction_features(self, features_raw: pd.DataFrame, encoded: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features (always included regardless of binning toggle)."""
        age = features_raw["age"].values
        hours = features_raw["hours-per-week"].values
        education = features_raw["education-num"].values
        capital_gain = features_raw["capital-gain"].values

        encoded = encoded.copy()
        encoded["age_x_hours"] = age * hours
        encoded["edu_x_capgain"] = education * capital_gain
        return encoded


    def _add_binned_features(self, features_raw: pd.DataFrame, encoded: pd.DataFrame) -> pd.DataFrame:
        """Add binned features from extensible list. Only called when use_binning=True."""
        hours = features_raw["hours-per-week"].values

        encoded = encoded.copy()
        # Bin hours-per-week: 0=part-time(<35), 1=full-time(35-45), 2=overtime(>45)
        encoded["hours_bin"] = np.where(hours < 35, 0, np.where(hours <= 45, 1, 2))

        # Add future bins here by appending to the DataFrame
        # e.g. encoded["capgain_nonzero"] = (features_raw["capital-gain"] > 0).astype(int)

        return encoded