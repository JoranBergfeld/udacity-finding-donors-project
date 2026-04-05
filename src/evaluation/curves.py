"""Evaluation curves: ROC, Precision-Recall, and Learning curves."""

import numpy as np
from sklearn.metrics import auc, fbeta_score, make_scorer, precision_recall_curve, roc_curve
from sklearn.model_selection import learning_curve


def compute_roc_curve(model, X_test, y_test) -> dict:
    """Compute ROC curve data for a fitted model.

    Returns dict with fpr, tpr, thresholds, and auc_score.
    """
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        raise ValueError(f"{model.__class__.__name__} has no predict_proba or decision_function")

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    auc_score = auc(fpr, tpr)

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(auc_score),
    }


def compute_pr_curve(model, X_test, y_test) -> dict:
    """Compute Precision-Recall curve data for a fitted model.

    Returns dict with precision, recall, thresholds, and auc_score.
    """
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        raise ValueError(f"{model.__class__.__name__} has no predict_proba or decision_function")

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    auc_score = auc(recall, precision)

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(auc_score),
    }


def compute_learning_curve(model, X, y, beta: float = 0.5, cv: int = 5, n_points: int = 10) -> dict:
    """Compute learning curve data (training score and validation score vs training size).

    Returns dict with train_sizes, train_scores_mean/std, val_scores_mean/std.
    """
    scorer = make_scorer(fbeta_score, beta=beta)
    train_sizes = np.linspace(0.1, 1.0, n_points)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv, scoring=scorer, random_state=42
    )

    return {
        "train_sizes": train_sizes_abs.tolist(),
        "train_scores_mean": train_scores.mean(axis=1).tolist(),
        "train_scores_std": train_scores.std(axis=1).tolist(),
        "val_scores_mean": val_scores.mean(axis=1).tolist(),
        "val_scores_std": val_scores.std(axis=1).tolist(),
    }
