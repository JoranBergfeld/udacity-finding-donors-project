"""Evaluation metrics: confusion matrix, naive predictor, cross-validation, comparison."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def naive_predictor(income: pd.Series, beta: float = 0.5) -> tuple[float, float, float, float]:
    """
    Compute metrics for a model that always predicts >50K (class 1).
    Returns (accuracy, recall, precision, fscore).
    """
    n = len(income) # Total samples equal to number of rows in income series
    true_positive = int(income.sum()) # Count of actual positives (income >50K) is the number of true positives for this naive predictor
    false_positive = n - true_positive # All other samples are false positives since we predict all as positive

    accuracy = true_positive / n # Accuracy is the proportion of correct predictions (true positives) out of all samples
    precision = true_positive / (true_positive + false_positive) # Precision is the proportion of true positives out of all predicted positives (which is all samples in this case)
    recall = 1.0 # Recall is 100% since we predict all as positive, so we catch all actual positives

    fscore = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) # F-beta score combines precision and recall with the given beta weight
    return accuracy, recall, precision, fscore


def compute_confusion_matrix(y_true, y_pred) -> dict:
    """
    Compute confusion matrix and return as labeled dict.
    Returns dict with keys: tp, fp, tn, fn, matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "matrix": cm,
    }


def compute_all_metrics(y_true, y_pred, beta: float = 0.5) -> dict:
    """
    Compute all Level 1-2 metrics for a single model prediction.
    Returns dict with confusion matrix counts + accuracy, precision, recall, fscore.
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    cm["accuracy"] = accuracy_score(y_true, y_pred)
    cm["precision"] = precision_score(y_true, y_pred, zero_division=0)
    cm["recall"] = recall_score(y_true, y_pred, zero_division=0)
    cm["fscore"] = fbeta_score(y_true, y_pred, beta=beta)
    return cm


def cross_validate_model(model, X, y, beta: float = 0.5, n_splits: int = 5) -> dict:
    """
    Run stratified k-fold cross-validation.
    Returns dict with mean/std for accuracy and F-beta, plus individual fold scores.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f_scorer = make_scorer(fbeta_score, beta=beta)

    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    f_scores = cross_val_score(model, X, y, cv=cv, scoring=f_scorer)

    return {
        "acc_mean": float(acc_scores.mean()),
        "acc_std": float(acc_scores.std()),
        "acc_scores": acc_scores.tolist(),
        "f_mean": float(f_scores.mean()),
        "f_std": float(f_scores.std()),
        "f_scores": f_scores.tolist(),
    }


def compare_configs(config_results: dict[str, dict]) -> pd.DataFrame:
    """
    Compare results across multiple configurations.
    Returns DataFrame with one row per config+model, sorted by f_test descending.
    """
    rows = []
    for config_name, models in config_results.items():
        for model_name, samples in models.items():
            metrics = samples[2]  # 100% training data
            rows.append(
                {
                    "config": config_name,
                    "model": model_name,
                    "acc_test": metrics["acc_test"],
                    "f_test": metrics["f_test"],
                    "train_time": metrics["train_time"],
                    "pred_time": metrics["pred_time"],
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values("f_test", ascending=False).reset_index(drop=True)
