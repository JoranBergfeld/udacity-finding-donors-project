"""Model training pipeline and configuration."""

from time import time

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.tree import DecisionTreeClassifier

# Single model config: 5 models (1 baseline + 4 tree-based contenders)
MODEL_CONFIGS = {
    "LogisticRegression": LogisticRegression(random_state=42, solver="liblinear"),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test, beta=0.5):
    """
    Train a model on a subset of the training data and make predictions on the test set.
    Returns dict with keys: train_time, acc_train, f_train, pred_time, acc_test, f_test.
    These values are used in the `evaluate()` function in visuals.py to create a performance comparison across models. 
    """
    results = {}

    start = time()
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()
    results["train_time"] = end - start

    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()
    results["pred_time"] = end - start

    results["acc_train"] = accuracy_score(y_train[:300], predictions_train)
    results["acc_test"] = accuracy_score(y_test, predictions_test)
    results["f_train"] = fbeta_score(y_train[:300], predictions_train, beta=beta)
    results["f_test"] = fbeta_score(y_test, predictions_test, beta=beta)

    return results


def run_all_models(models, X_train, y_train, X_test, y_test, beta=0.5):
    """Run all models at 1%, 10%, 100% sample sizes.

    Returns dict matching visuals.evaluate() format:
    {"ModelName": {0: {...}, 1: {...}, 2: {...}}}
    """
    samples_100 = len(y_train)
    samples_10 = int(samples_100 * 0.1)
    samples_1 = int(samples_100 * 0.01)
    sample_sizes = [samples_1, samples_10, samples_100]

    results = {}
    for name, clf in models.items():
        results[name] = {}
        for i, sample_size in enumerate(sample_sizes):
            results[name][i] = train_predict(
                clf, sample_size, X_train, y_train, X_test, y_test, beta=beta
            )

    return results
