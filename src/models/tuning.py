"""Hyperparameter tuning utilities."""

from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV

PARAM_GRIDS = {
    "GradientBoostingClassifier": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1, 0.2],
    },
    "RandomForestClassifier": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
    },
    "AdaBoostClassifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.5, 1.0, 2.0],
    },
    "LogisticRegression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "liblinear"],
    },
    "DecisionTreeClassifier": {
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
    },
}


def grid_search_tune(clf, param_grid, X_train, y_train, beta=0.5, cv=5):
    """
    Run GridSearchCV with F-beta scoring.
    Returns the fitted GridSearchCV object.
    """
    scorer = make_scorer(fbeta_score, beta=beta)
    grid_object = GridSearchCV(clf, param_grid, scoring=scorer, cv=cv)
    grid_fit = grid_object.fit(X_train, y_train)
    return grid_fit
