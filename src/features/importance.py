"""Feature importance extraction and selection."""

import numpy as np
import pandas as pd


def extract_importances(model_class, X_train, y_train):
    """
    Fit a model and return (fitted_model, importances_array).
    The model class must support `feature_importances_` after fitting.
    """
    model = model_class(random_state=42) if callable(model_class) else model_class
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    return model, importances


def get_top_k_features(importances: np.ndarray, X_train: pd.DataFrame, k: int = 5) -> list[str]:
    """Return names of top-k most important features."""
    indices = np.argsort(importances)[::-1][:k]
    return X_train.columns.values[indices].tolist()


def reduce_features(X_train: pd.DataFrame, X_test: pd.DataFrame, importances: np.ndarray, k: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce datasets to only top-k features by importance."""
    top_features = get_top_k_features(importances, X_train, k)
    return X_train[top_features], X_test[top_features]
