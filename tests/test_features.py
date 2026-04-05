"""Tests for src.features.importance."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from src.features.importance import extract_importances, get_top_k_features, reduce_features


def _make_data():
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame(rng.randn(n, 10), columns=[f"f{i}" for i in range(10)])
    y = pd.Series(rng.randint(0, 2, n))
    return X[:160], X[160:], y[:160], y[160:]


def test_extract_importances_length():
    X_train, _, y_train, _ = _make_data()
    _, importances = extract_importances(GradientBoostingClassifier, X_train, y_train)
    assert len(importances) == X_train.shape[1]


def test_extract_importances_sum_to_one():
    X_train, _, y_train, _ = _make_data()
    _, importances = extract_importances(GradientBoostingClassifier, X_train, y_train)
    assert abs(importances.sum() - 1.0) < 1e-6


def test_get_top_k_returns_k_features():
    X_train, _, y_train, _ = _make_data()
    _, importances = extract_importances(GradientBoostingClassifier, X_train, y_train)
    top = get_top_k_features(importances, X_train, k=5)
    assert len(top) == 5
    assert all(f in X_train.columns for f in top)


def test_reduce_features_shape():
    X_train, X_test, y_train, _ = _make_data()
    _, importances = extract_importances(GradientBoostingClassifier, X_train, y_train)
    X_train_r, X_test_r = reduce_features(X_train, X_test, importances, k=3)
    assert X_train_r.shape[1] == 3
    assert X_test_r.shape[1] == 3
    assert list(X_train_r.columns) == list(X_test_r.columns)
