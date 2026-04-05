"""Tests for src.models.training."""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from src.models.training import (
    MODEL_CONFIGS,
    run_all_models,
    train_predict,
)


def _make_tiny_data():
    """Create a tiny dataset for fast model tests."""
    rng = np.random.RandomState(42)
    n = 500
    X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.randint(0, 2, n))
    return X[:400], X[400:], y[:400], y[400:]


def test_train_predict_returns_all_keys():
    X_train, X_test, y_train, y_test = _make_tiny_data()
    clf = DecisionTreeClassifier(random_state=42)
    result = train_predict(clf, 100, X_train, y_train, X_test, y_test)
    expected_keys = {"train_time", "acc_train", "f_train", "pred_time", "acc_test", "f_test"}
    assert set(result.keys()) == expected_keys


def test_train_predict_scores_in_range():
    X_train, X_test, y_train, y_test = _make_tiny_data()
    clf = DecisionTreeClassifier(random_state=42)
    result = train_predict(clf, 100, X_train, y_train, X_test, y_test)
    for key in ["acc_train", "acc_test", "f_train", "f_test"]:
        assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"


def test_train_predict_times_positive():
    X_train, X_test, y_train, y_test = _make_tiny_data()
    clf = DecisionTreeClassifier(random_state=42)
    result = train_predict(clf, 100, X_train, y_train, X_test, y_test)
    assert result["train_time"] >= 0
    assert result["pred_time"] >= 0


def test_run_all_models_structure():
    X_train, X_test, y_train, y_test = _make_tiny_data()
    models = {"DT": DecisionTreeClassifier(random_state=42)}
    results = run_all_models(models, X_train, y_train, X_test, y_test)
    assert "DT" in results
    assert set(results["DT"].keys()) == {0, 1, 2}
    for i in range(3):
        assert "acc_test" in results["DT"][i]
        assert "f_test" in results["DT"][i]


def test_model_configs_has_five_models():
    assert len(MODEL_CONFIGS) == 5
    expected = {
        "LogisticRegression",
        "DecisionTree",
        "RandomForest",
        "AdaBoost",
        "GradientBoosting",
    }
    assert set(MODEL_CONFIGS.keys()) == expected


def test_model_configs_have_random_state():
    for name, model in MODEL_CONFIGS.items():
        if hasattr(model, "random_state"):
            assert model.random_state is not None, f"{name} missing random_state"
