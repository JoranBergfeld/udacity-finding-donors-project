"""Tests for src.evaluation.metrics and src.evaluation.curves."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from src.evaluation.curves import compute_learning_curve, compute_pr_curve, compute_roc_curve
from src.evaluation.metrics import (
    compare_configs,
    compute_all_metrics,
    compute_confusion_matrix,
    cross_validate_model,
    naive_predictor,
)

# --- Naive Predictor ---


def test_naive_predictor_recall_is_one():
    income = pd.Series([0, 0, 0, 1, 1])
    _, recall, _, _ = naive_predictor(income)
    assert recall == 1.0


def test_naive_predictor_accuracy():
    income = pd.Series([0, 0, 0, 1, 1])
    accuracy, _, _, _ = naive_predictor(income)
    assert accuracy == 2 / 5


def test_naive_predictor_precision_equals_accuracy():
    income = pd.Series([0, 0, 0, 1, 1])
    accuracy, _, precision, _ = naive_predictor(income)
    assert accuracy == precision


def test_naive_predictor_fscore_formula():
    income = pd.Series([0, 0, 1, 1, 1])
    accuracy, recall, precision, fscore = naive_predictor(income, beta=0.5)
    expected = (1 + 0.25) * (precision * recall) / (0.25 * precision + recall)
    assert abs(fscore - expected) < 1e-10


# --- Confusion Matrix ---


def test_confusion_matrix_counts():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm["tp"] == 2
    assert cm["fn"] == 1
    assert cm["fp"] == 1
    assert cm["tn"] == 1


def test_compute_all_metrics_keys():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    metrics = compute_all_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "fscore" in metrics
    assert "tp" in metrics


def test_compute_all_metrics_range():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0, 1, 1])
    metrics = compute_all_metrics(y_true, y_pred)
    for key in ["accuracy", "precision", "recall", "fscore"]:
        assert 0.0 <= metrics[key] <= 1.0


# --- Cross-validation ---


def _make_tiny_data():
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.randint(0, 2, n))
    return X, y


def test_cross_validate_returns_keys():
    X, y = _make_tiny_data()
    result = cross_validate_model(GradientBoostingClassifier(random_state=42), X, y, n_splits=3)
    assert "acc_mean" in result
    assert "acc_std" in result
    assert "f_mean" in result
    assert "f_std" in result
    assert len(result["acc_scores"]) == 3


def test_cross_validate_scores_in_range():
    X, y = _make_tiny_data()
    result = cross_validate_model(GradientBoostingClassifier(random_state=42), X, y, n_splits=3)
    assert 0.0 <= result["acc_mean"] <= 1.0
    assert 0.0 <= result["f_mean"] <= 1.0
    assert result["acc_std"] >= 0


# --- ROC / PR curves ---


def test_roc_curve_keys():
    X, y = _make_tiny_data()
    model = GradientBoostingClassifier(random_state=42).fit(X, y)
    roc = compute_roc_curve(model, X, y)
    assert "fpr" in roc
    assert "tpr" in roc
    assert "auc" in roc
    assert 0.0 <= roc["auc"] <= 1.0


def test_pr_curve_keys():
    X, y = _make_tiny_data()
    model = GradientBoostingClassifier(random_state=42).fit(X, y)
    pr = compute_pr_curve(model, X, y)
    assert "precision" in pr
    assert "recall" in pr
    assert "auc" in pr
    assert 0.0 <= pr["auc"] <= 1.0


# --- Learning curves ---


def test_learning_curve_keys():
    X, y = _make_tiny_data()
    lc = compute_learning_curve(GradientBoostingClassifier(random_state=42), X, y, cv=3, n_points=5)
    assert "train_sizes" in lc
    assert "train_scores_mean" in lc
    assert "val_scores_mean" in lc
    assert len(lc["train_sizes"]) == 5


# --- Compare configs ---


def test_compare_configs():
    config_results = {
        "config_1": {
            "ModelA": {
                0: {"acc_test": 0.7, "f_test": 0.6, "train_time": 0.1, "pred_time": 0.01},
                1: {"acc_test": 0.75, "f_test": 0.65, "train_time": 0.2, "pred_time": 0.02},
                2: {"acc_test": 0.8, "f_test": 0.7, "train_time": 0.3, "pred_time": 0.03},
            }
        }
    }
    df = compare_configs(config_results)
    assert len(df) == 1
    assert df.iloc[0]["f_test"] == 0.7
