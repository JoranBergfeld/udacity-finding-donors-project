"""Run 10-run experiment grid (5 models x 2 binning states) with 5-level evaluation.

Groups:
  Group A: CensusPipeline(use_binning=False) -> 5 models
  Group B: CensusPipeline(use_binning=True)  -> 5 models
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from src.data.loader import load_census
from src.data.preprocessing import CensusPipeline
from src.evaluation.curves import compute_learning_curve, compute_pr_curve, compute_roc_curve
from src.evaluation.metrics import (
    compute_all_metrics,
    cross_validate_model,
    naive_predictor,
)
from src.models.training import MODEL_CONFIGS, run_all_models


def run_group(group_name, pipeline, models, data):
    """Run full evaluation for one binning group."""
    features_raw = data.drop("income", axis=1)
    income_raw = data["income"]

    features_final, income = pipeline.fit_transform(features_raw, income_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        features_final, income, test_size=0.2, random_state=0
    )

    print(f"\n{'=' * 80}")
    print(f"  {group_name} ({features_final.shape[1]} features)")
    print(f"{'=' * 80}")

    # --- Level 2: Basic metrics at 1%/10%/100% ---
    fresh_models = {name: clone(clf) for name, clf in models.items()}
    results = run_all_models(fresh_models, X_train, y_train, X_test, y_test)

    print("\n--- Level 2: Metrics at 100% training data ---")
    for name, samples in results.items():
        m = samples[2]
        print(
            f"  {name:25s}  F-0.5={m['f_test']:.4f}"
            f"  Acc={m['acc_test']:.4f}  Time={m['train_time']:.3f}s"
        )

    # --- Level 1: Confusion matrices ---
    print("\n--- Level 1: Confusion matrices ---")
    cm_results = {}
    for name, clf in models.items():
        model = clone(clf).fit(X_train, y_train)
        preds = model.predict(X_test)
        cm = compute_all_metrics(y_test, preds)
        cm_results[name] = cm
        print(
            f"  {name:25s}  TP={cm['tp']:5d}  FP={cm['fp']:5d}"
            f"  TN={cm['tn']:5d}  FN={cm['fn']:5d}"
            f"  Prec={cm['precision']:.4f}  Rec={cm['recall']:.4f}"
        )

    # --- Level 3: ROC/PR curves (AUC only in console) ---
    print("\n--- Level 3: ROC and PR curve AUCs ---")
    for name, clf in models.items():
        model = clone(clf).fit(X_train, y_train)
        roc = compute_roc_curve(model, X_test, y_test)
        pr = compute_pr_curve(model, X_test, y_test)
        print(f"  {name:25s}  AUC-ROC={roc['auc']:.4f}  AUC-PR={pr['auc']:.4f}")

    # --- Level 5: Cross-validation ---
    print("\n--- Level 5: 5-Fold Cross-Validation ---")
    cv_results = {}
    for name, clf in models.items():
        cv = cross_validate_model(clone(clf), features_final, income, n_splits=5)
        cv_results[name] = cv
        print(
            f"  {name:25s}  F-0.5={cv['f_mean']:.4f} +/- {cv['f_std']:.4f}"
            f"  Acc={cv['acc_mean']:.4f} +/- {cv['acc_std']:.4f}"
        )

    return {
        "results": results,
        "cm": cm_results,
        "cv": cv_results,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def main():
    data = load_census("starter/census.csv")
    accuracy, _, _, fscore = naive_predictor(
        data["income"].map(lambda x: 1 if x.strip() == ">50K" else 0)
    )
    print(f"Naive predictor baseline:  Accuracy={accuracy:.4f}  F-0.5={fscore:.4f}")

    # Run both groups
    group_a = run_group(
        "Group A (no binning)",
        CensusPipeline(use_binning=False),
        MODEL_CONFIGS,
        data,
    )
    group_b = run_group(
        "Group B (with binning)",
        CensusPipeline(use_binning=True),
        MODEL_CONFIGS,
        data,
    )

    # --- Cross-group comparison ---
    print(f"\n{'=' * 80}")
    print("  CROSS-GROUP: Binning effect per model (CV F-0.5)")
    print(f"{'=' * 80}")
    print(f"  {'Model':25s}  {'No Binning':>12s}  {'With Binning':>12s}  {'Delta':>8s}")
    print(f"  {'-' * 25}  {'-' * 12}  {'-' * 12}  {'-' * 8}")

    for name in MODEL_CONFIGS:
        f_a = group_a["cv"][name]["f_mean"]
        f_b = group_b["cv"][name]["f_mean"]
        delta = f_b - f_a
        direction = "+" if delta >= 0 else ""
        print(f"  {name:25s}  {f_a:12.4f}  {f_b:12.4f}  {direction}{delta:.4f}")

    # Overall best
    best_name = None
    best_f = -1
    best_group = None
    for name in MODEL_CONFIGS:
        for group_label, group in [("no_binning", group_a), ("with_binning", group_b)]:
            f = group["cv"][name]["f_mean"]
            if f > best_f:
                best_f = f
                best_name = name
                best_group = group_label

    print(f"\nOverall best: {best_name} ({best_group}) with CV F-0.5 = {best_f:.4f}")
    print("This model should proceed to GridSearchCV tuning.")

    # --- Level 4: Learning curves for top model only (slow) ---
    print(f"\n--- Level 4: Learning curve for {best_name} (no binning) ---")
    from sklearn.base import clone as sk_clone

    lc = compute_learning_curve(
        sk_clone(MODEL_CONFIGS[best_name]),
        group_a["X_train"],
        group_a["y_train"],
        cv=3,
        n_points=5,
    )
    for size, train_s, val_s in zip(
        lc["train_sizes"], lc["train_scores_mean"], lc["val_scores_mean"], strict=True
    ):
        print(f"  n={size:6d}  Train F-0.5={train_s:.4f}  Val F-0.5={val_s:.4f}")


if __name__ == "__main__":
    main()
