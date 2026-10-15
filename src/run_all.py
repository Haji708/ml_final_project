"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all

This orchestrator should:
  1. load + clean the data,
  2. build features, make the splits, standardize (for the SVM),
  3. train YOUR decision tree (Task A regression, Task B classification),
  4. train YOUR Pegasos SVM (Task B classification),
  5. train the scikit-learn baselines with matched settings,
  6. compute metrics and SAVE all figures to report/figures/,
  7. print a final comparison table.

Everything must be deterministic (fixed seeds). Keep print output tidy so a
grader can read the headline numbers straight from the console.
"""

from __future__ import annotations

from .member1_regression import run_member1_pipeline
# from .decision_tree import DecisionTree
# from .svm import PegasosSVM
# from . import evaluate


def main() -> None:
    # 1. Member 1: data pipeline and Task A regression --------------------
    prepared = run_member1_pipeline(
        data_path="data/bina_az_sale.csv"
    )

    X_train = prepared["X_train"]
    X_validation = prepared["X_validation"]
    X_test = prepared["X_test"]

    X_train_scaled = prepared["X_train_scaled"]
    X_validation_scaled = prepared["X_validation_scaled"]
    X_test_scaled = prepared["X_test_scaled"]

    y_train = prepared["y_train"]
    y_validation = prepared["y_validation"]
    y_test = prepared["y_test"]

    y_tier_train = prepared["y_tier_train"]
    y_tier_validation = prepared["y_tier_validation"]
    y_tier_test = prepared["y_tier_test"]

    # 2. Decision-tree member --------------------------------------------
    # Trees should use the unscaled X arrays.

    # 3. SVM member ------------------------------------------------------
    # SVM should use X_train_scaled, X_validation_scaled, X_test_scaled.

    # 4. Shared figures and final comparison -----------------------------


if __name__ == "__main__":
    main()
