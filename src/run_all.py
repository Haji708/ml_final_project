"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all

This orchestrator:
  1. Loads and cleans the bina.az dataset with zero target/test leakage,
  2. Builds features, stratified splits, and standardized matrices,
  3. Trains Task A regression models (from-scratch Ridge, sklearn Ridge, DecisionTreeRegressor),
  4. Trains Task B classification models (from-scratch PegasosSVM, sklearn SGDClassifier, LinearSVC, DecisionTreeClassifier),
  5. Evaluates all models using our from-scratch src.evaluate suite,
  6. Generates error analysis and model interpretation figures into report/figures/,
  7. Prints comprehensive comparison tables and saves results to report/summary_table.csv.

All runs are seeded (SEED = 42) for 100% deterministic reproducibility.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import data_prep
from . import evaluate
from .linear_regression import RidgeRegressionNormalEquation
from .svm import PegasosSVM

# Optional from-scratch decision tree
try:
    from .decision_tree import DecisionTree
except ImportError:
    DecisionTree = None

from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


FIG_DIR = Path("report/figures")
REPORT_DIR = Path("report")


def plot_svm_weights(feature_names: list[str], weights: np.ndarray, save_path: Path) -> None:
    """Plot top positive and negative features learned by the Pegasos SVM."""
    feat_arr = np.array(feature_names)
    pos_idx = np.argsort(weights)[-10:]
    neg_idx = np.argsort(weights)[:10]
    sel_idx = np.concatenate([neg_idx, pos_idx])

    sel_names = [feat_arr[i] for i in sel_idx]
    sel_weights = weights[sel_idx]
    colors = ["#e74c3c" if w < 0 else "#2ecc71" for w in sel_weights]

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(sel_weights))
    plt.barh(y_pos, sel_weights, color=colors, edgecolor="black", alpha=0.85)
    plt.yticks(y_pos, sel_names, fontsize=9)
    plt.axvline(0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("SVM Learned Weight (w)", fontsize=11)
    plt.title("Top Feature Weights — From-Scratch Pegasos SVM (Task B)", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tree_feature_importances(feature_names: list[str], importances: np.ndarray, save_path: Path) -> None:
    """Plot top impurity-decrease feature importances from decision tree."""
    top_indices = np.argsort(importances)[-15:]
    top_names = [feature_names[i] for i in top_indices]
    top_imps = importances[top_indices]

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(top_imps))
    plt.barh(y_pos, top_imps, color="#3498db", edgecolor="black", alpha=0.85)
    plt.yticks(y_pos, top_names, fontsize=9)
    plt.xlabel("MDI Feature Importance", fontsize=11)
    plt.title("Top 15 Feature Importances — DecisionTree", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_error_analysis(
    y_true_reg: np.ndarray,
    y_pred_reg: np.ndarray,
    y_true_clf: np.ndarray,
    y_pred_clf: np.ndarray,
    save_path: Path,
) -> None:
    """Generate diagnostic error plots for regression residuals and classification errors."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Regression Residual Plot
    residuals = y_true_reg - y_pred_reg
    axes[0].scatter(y_pred_reg / 1e3, residuals / 1e3, alpha=0.25, s=12, color="#2980b9")
    axes[0].axhline(0, color="crimson", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted Price (k AZN)", fontsize=11)
    axes[0].set_ylabel("Residual (True - Predicted) (k AZN)", fontsize=11)
    axes[0].set_title("Task A: Regression Residuals vs Predicted", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].set_xlim(left=0, right=min(np.percentile(y_pred_reg / 1e3, 99.5), 3000))

    # 2. Classification Confusion Matrix Heatmap
    cm = evaluate.confusion_matrix(y_true_clf, y_pred_clf)
    im = axes[1].imshow(cm, cmap="Blues", interpolation="nearest")
    axes[1].set_title("Task B: Pegasos SVM Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    tick_marks = [0, 1]
    axes[1].set_xticks(tick_marks)
    axes[1].set_xticklabels(["Standard (0)", "Premium (1)"])
    axes[1].set_yticks(tick_marks)
    axes[1].set_yticklabels(["Standard (0)", "Premium (1)"])
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)

    for i in range(2):
        for j in range(2):
            axes[1].text(
                j, i, f"{cm[i, j]:,}",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=13,
                fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def select_svm_config(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """Select SVM hyperparameters using validation F1 only."""
    candidates = [
        {
            "feature_map": "linear",
            "lambda_": 1e-5,
            "n_components": 500,
            "gamma": 1.0,
        },
        {
            "feature_map": "linear",
            "lambda_": 1e-4,
            "n_components": 500,
            "gamma": 1.0,
        },
        {
            "feature_map": "linear",
            "lambda_": 1e-3,
            "n_components": 500,
            "gamma": 1.0,
        },
        {
            "feature_map": "rff",
            "lambda_": 1e-4,
            "n_components": 250,
            "gamma": 0.5,
        },
        {
            "feature_map": "rff",
            "lambda_": 1e-4,
            "n_components": 500,
            "gamma": 1.0,
        },
        {
            "feature_map": "rff",
            "lambda_": 1e-4,
            "n_components": 500,
            "gamma": 2.0,
        },
    ]

    trials = []

    for config in candidates:
        model = PegasosSVM(
            lambda_=config["lambda_"],
            n_iters=50_000,
            random_state=42,
            feature_map=config["feature_map"],
            n_components=config["n_components"],
            gamma=config["gamma"],
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)
        scores = model.decision_function(X_validation)

        precision, recall, f1 = evaluate.precision_recall_f1(
            y_validation,
            predictions,
        )
        auc = evaluate.roc_auc(y_validation, scores)

        trials.append({
            **config,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUC": auc,
        })

    trials_df = pd.DataFrame(trials)
    best_config = trials_df.sort_values(
        ["F1", "AUC"],
        ascending=False,
    ).iloc[0].to_dict()

    return best_config, trials_df

def main() -> None:
    print("=" * 80)
    print("  MLE-AI-201 FINAL PROJECT: REPRODUCIBLE BENCHMARK PIPELINE")
    print("=" * 80)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Data Pipeline
    # -------------------------------------------------------------------------
    print("\n[Step 1/4] Preparing data pipeline (clean, encode, stratify, scale)...")
    t0 = time.time()
    bundle = data_prep.prepare_data(seed=42)
    prep_time = time.time() - t0

    X_train = bundle["X_train"]
    X_val = bundle["X_validation"]
    X_test = bundle["X_test"]

    X_train_sc = bundle["X_train_scaled"]
    X_val_sc = bundle["X_validation_scaled"]
    X_test_sc = bundle["X_test_scaled"]

    y_train = bundle["y_train"]
    y_val = bundle["y_validation"]
    y_test = bundle["y_test"]

    y_tier_tr = bundle["y_tier_train"]
    y_tier_val = bundle["y_tier_validation"]
    y_tier_test = bundle["y_tier_test"]

    feat_names = bundle["feature_names"]
    threshold = bundle["tier_threshold"]

    print(f"  ✓ Data ready in {prep_time:.2f}s")
    print(f"  ✓ Train size: {len(X_train):,}, Val size: {len(X_val):,}, Test size: {len(X_test):,}")
    print(f"  ✓ Features count: {X_train.shape[1]}")
    print(f"  ✓ Derived Price Tier Median Threshold (train only): {threshold:,.0f} AZN")

    val_results: dict[str, dict[str, float]] = {}
    test_results: dict[str, dict[str, float]] = {}

    # -------------------------------------------------------------------------
    # 2. Task A: Regression (Price Prediction)
    # -------------------------------------------------------------------------
    print("\n[Step 2/4] Training and evaluating Task A regression models...")

    # Model A1: Normal Equation Ridge Regression (From Scratch)
    print("  -> Training From-Scratch Ridge Regression (Normal Equation, alpha=10.0)...")
    t_start = time.time()
    ridge_scratch = RidgeRegressionNormalEquation(alpha=10.0)
    ridge_scratch.fit(X_train_sc, y_train)
    t_fit_ridge_sc = time.time() - t_start
    preds_val_r_sc = ridge_scratch.predict(X_val_sc)
    preds_test_r_sc = ridge_scratch.predict(X_test_sc)

    val_results["Ridge (Scratch)"] = {
        "RMSE": evaluate.rmse(y_val, preds_val_r_sc),
        "MAE": evaluate.mae(y_val, preds_val_r_sc),
        "R2": evaluate.r2(y_val, preds_val_r_sc),
    }
    test_results["Ridge (Scratch)"] = {
        "RMSE": evaluate.rmse(y_test, preds_test_r_sc),
        "MAE": evaluate.mae(y_test, preds_test_r_sc),
        "R2": evaluate.r2(y_test, preds_test_r_sc),
    }
    print(f"     Val R2: {val_results['Ridge (Scratch)']['R2']:.4f} (fit: {t_fit_ridge_sc:.2f}s)")

    # Model A2: Scikit-learn Ridge Baseline
    print("  -> Training scikit-learn Ridge Baseline (alpha=10.0)...")
    t_start = time.time()
    ridge_sk = Ridge(alpha=10.0, random_state=42)
    ridge_sk.fit(X_train_sc, y_train)
    t_fit_ridge_sk = time.time() - t_start
    preds_val_r_sk = ridge_sk.predict(X_val_sc)
    preds_test_r_sk = ridge_sk.predict(X_test_sc)

    val_results["Ridge (sklearn)"] = {
        "RMSE": evaluate.rmse(y_val, preds_val_r_sk),
        "MAE": evaluate.mae(y_val, preds_val_r_sk),
        "R2": evaluate.r2(y_val, preds_val_r_sk),
    }
    test_results["Ridge (sklearn)"] = {
        "RMSE": evaluate.rmse(y_test, preds_test_r_sk),
        "MAE": evaluate.mae(y_test, preds_test_r_sk),
        "R2": evaluate.r2(y_test, preds_test_r_sk),
    }
    print(f"     Val R2: {val_results['Ridge (sklearn)']['R2']:.4f} (fit: {t_fit_ridge_sk:.2f}s)")

    # Model A3: Scikit-learn DecisionTreeRegressor Baseline
    print("  -> Training scikit-learn DecisionTreeRegressor Baseline (max_depth=12)...")
    t_start = time.time()
    dt_reg = DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=42)
    dt_reg.fit(X_train, y_train)
    t_fit_dt_reg = time.time() - t_start
    preds_val_dt_reg = dt_reg.predict(X_val)
    preds_test_dt_reg = dt_reg.predict(X_test)

    val_results["DecisionTree Reg (sklearn)"] = {
        "RMSE": evaluate.rmse(y_val, preds_val_dt_reg),
        "MAE": evaluate.mae(y_val, preds_val_dt_reg),
        "R2": evaluate.r2(y_val, preds_val_dt_reg),
    }
    test_results["DecisionTree Reg (sklearn)"] = {
        "RMSE": evaluate.rmse(y_test, preds_test_dt_reg),
        "MAE": evaluate.mae(y_test, preds_test_dt_reg),
        "R2": evaluate.r2(y_test, preds_test_dt_reg),
    }
    print(f"     Val R2: {val_results['DecisionTree Reg (sklearn)']['R2']:.4f} (fit: {t_fit_dt_reg:.2f}s)")

    # Model A4: From-scratch DecisionTree (if implemented)
    if DecisionTree is not None:
        try:
            tree_reg = DecisionTree(task="regression", criterion="mse", max_depth=10, random_state=42)
            tree_reg.fit(X_train, y_train)
            p_val = tree_reg.predict(X_val)
            p_test = tree_reg.predict(X_test)
            val_results["DecisionTree Reg (Scratch)"] = {
                "RMSE": evaluate.rmse(y_val, p_val),
                "MAE": evaluate.mae(y_val, p_val),
                "R2": evaluate.r2(y_val, p_val),
            }
            test_results["DecisionTree Reg (Scratch)"] = {
                "RMSE": evaluate.rmse(y_test, p_test),
                "MAE": evaluate.mae(y_test, p_test),
                "R2": evaluate.r2(y_test, p_test),
            }
        except (NotImplementedError, Exception):
            pass

    # -------------------------------------------------------------------------
    # 3. Task B: Classification (Price Tier)
    # -------------------------------------------------------------------------
    print("\n[Step 3/4] Training and evaluating Task B classification models...")

    print("  -> Selecting SVM configuration on validation data...")

    best_svm_config, svm_trials = select_svm_config(
        X_train_sc,
        y_tier_tr,
        X_val_sc,
        y_tier_val,
    )

    svm_trials.to_csv(
        REPORT_DIR / "svm_validation_trials.csv",
        index=False,
    )

    print(
        "     Selected:",
        {
            key: best_svm_config[key]
            for key in (
                "feature_map",
                "lambda_",
                "gamma",
                "n_components",
            )
        },
        f"with validation F1={best_svm_config['F1']:.4f}",
    )

    # Model B1: validation-selected From-Scratch Pegasos Soft-Margin SVM
    print("  -> Training validation-selected From-Scratch Pegasos SVM...")
    t_start = time.time()
    pegasos_svm = PegasosSVM(
        lambda_=float(best_svm_config["lambda_"]),
        n_iters=50_000,
        random_state=42,
        feature_map=str(best_svm_config["feature_map"]),
        n_components=int(best_svm_config["n_components"]),
        gamma=float(best_svm_config["gamma"]),
        record_every=100,
    )
    pegasos_svm.fit(X_train_sc, y_tier_tr)
    t_fit_svm = time.time() - t_start

    preds_val_svm = pegasos_svm.predict(X_val_sc)
    scores_val_svm = pegasos_svm.decision_function(X_val_sc)
    preds_test_svm = pegasos_svm.predict(X_test_sc)
    scores_test_svm = pegasos_svm.decision_function(X_test_sc)

    p_v, r_v, f1_v = evaluate.precision_recall_f1(y_tier_val, preds_val_svm)
    auc_v = evaluate.roc_auc(y_tier_val, scores_val_svm)
    val_results["PegasosSVM (Scratch)"] = {"Precision": p_v, "Recall": r_v, "F1": f1_v, "AUC": auc_v}

    p_t, r_t, f1_t = evaluate.precision_recall_f1(y_tier_test, preds_test_svm)
    auc_t = evaluate.roc_auc(y_tier_test, scores_test_svm)
    test_results["PegasosSVM (Scratch)"] = {"Precision": p_t, "Recall": r_t, "F1": f1_t, "AUC": auc_t}
    print(f"     Val F1: {f1_v:.4f}, AUC: {auc_v:.4f} (fit: {t_fit_svm:.2f}s)")

    # Model B1b: From-Scratch Pegasos SVM with Random Fourier Features
    print("  -> Training From-Scratch Pegasos SVM with RFF...")
    t_start = time.time()

    pegasos_rff = PegasosSVM(
        lambda_=1e-4,
        n_iters=50_000,
        random_state=42,
        feature_map="rff",
        n_components=500,
        gamma=1.0,
        record_every=100,
    )
    pegasos_rff.fit(X_train_sc, y_tier_tr)

    t_fit_rff = time.time() - t_start

    t_predict = time.time()
    preds_val_rff = pegasos_rff.predict(X_val_sc)
    scores_val_rff = pegasos_rff.decision_function(X_val_sc)
    preds_test_rff = pegasos_rff.predict(X_test_sc)
    scores_test_rff = pegasos_rff.decision_function(X_test_sc)
    t_predict_rff = time.time() - t_predict

    p_v, r_v, f1_v = evaluate.precision_recall_f1(
        y_tier_val,
        preds_val_rff,
    )
    auc_v = evaluate.roc_auc(y_tier_val, scores_val_rff)

    val_results["PegasosSVM (RFF)"] = {
        "Precision": p_v,
        "Recall": r_v,
        "F1": f1_v,
        "AUC": auc_v,
    }

    p_t, r_t, f1_t = evaluate.precision_recall_f1(
        y_tier_test,
        preds_test_rff,
    )
    auc_t = evaluate.roc_auc(y_tier_test, scores_test_rff)

    test_results["PegasosSVM (RFF)"] = {
        "Precision": p_t,
        "Recall": r_t,
        "F1": f1_t,
        "AUC": auc_t,
    }

    print(
        f"     Val F1: {f1_v:.4f}, "
        f"AUC: {auc_v:.4f}, "
        f"fit: {t_fit_rff:.2f}s, "
        f"predict: {t_predict_rff:.2f}s"
    )

    # Model B2: Scikit-learn SGDClassifier (hinge loss baseline)
    print("  -> Training scikit-learn SGDClassifier (hinge loss baseline)...")
    t_start = time.time()
    sgd_clf = SGDClassifier(loss="hinge", alpha=1e-4, random_state=42)
    sgd_clf.fit(X_train_sc, y_tier_tr)
    t_fit_sgd = time.time() - t_start

    preds_val_sgd = sgd_clf.predict(X_val_sc)
    scores_val_sgd = sgd_clf.decision_function(X_val_sc)
    preds_test_sgd = sgd_clf.predict(X_test_sc)
    scores_test_sgd = sgd_clf.decision_function(X_test_sc)

    p_v, r_v, f1_v = evaluate.precision_recall_f1(y_tier_val, preds_val_sgd)
    auc_v = evaluate.roc_auc(y_tier_val, scores_val_sgd)
    val_results["SGDClassifier (hinge)"] = {"Precision": p_v, "Recall": r_v, "F1": f1_v, "AUC": auc_v}

    p_t, r_t, f1_t = evaluate.precision_recall_f1(y_tier_test, preds_test_sgd)
    auc_t = evaluate.roc_auc(y_tier_test, scores_test_sgd)
    test_results["SGDClassifier (hinge)"] = {"Precision": p_t, "Recall": r_t, "F1": f1_t, "AUC": auc_t}
    print(f"     Val F1: {f1_v:.4f}, AUC: {auc_v:.4f} (fit: {t_fit_sgd:.2f}s)")

    # Model B3: Scikit-learn LinearSVC Baseline
    print("  -> Training scikit-learn LinearSVC Baseline (C=1.0)...")
    t_start = time.time()
    lsvc = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    lsvc.fit(X_train_sc, y_tier_tr)
    t_fit_lsvc = time.time() - t_start

    preds_val_lsvc = lsvc.predict(X_val_sc)
    scores_val_lsvc = lsvc.decision_function(X_val_sc)
    preds_test_lsvc = lsvc.predict(X_test_sc)
    scores_test_lsvc = lsvc.decision_function(X_test_sc)

    p_v, r_v, f1_v = evaluate.precision_recall_f1(y_tier_val, preds_val_lsvc)
    auc_v = evaluate.roc_auc(y_tier_val, scores_val_lsvc)
    val_results["LinearSVC (sklearn)"] = {"Precision": p_v, "Recall": r_v, "F1": f1_v, "AUC": auc_v}

    p_t, r_t, f1_t = evaluate.precision_recall_f1(y_tier_test, preds_test_lsvc)
    auc_t = evaluate.roc_auc(y_tier_test, scores_test_lsvc)
    test_results["LinearSVC (sklearn)"] = {"Precision": p_t, "Recall": r_t, "F1": f1_t, "AUC": auc_t}
    print(f"     Val F1: {f1_v:.4f}, AUC: {auc_v:.4f} (fit: {t_fit_lsvc:.2f}s)")

    # Model B4: Scikit-learn DecisionTreeClassifier Baseline
    print("  -> Training scikit-learn DecisionTreeClassifier Baseline (max_depth=12)...")
    t_start = time.time()
    dt_clf = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=42)
    dt_clf.fit(X_train, y_tier_tr)
    t_fit_dt_clf = time.time() - t_start

    preds_val_dt_clf = dt_clf.predict(X_val)
    scores_val_dt_clf = dt_clf.predict_proba(X_val)[:, 1]
    preds_test_dt_clf = dt_clf.predict(X_test)
    scores_test_dt_clf = dt_clf.predict_proba(X_test)[:, 1]

    p_v, r_v, f1_v = evaluate.precision_recall_f1(y_tier_val, preds_val_dt_clf)
    auc_v = evaluate.roc_auc(y_tier_val, scores_val_dt_clf)
    val_results["DecisionTree Clf (sklearn)"] = {"Precision": p_v, "Recall": r_v, "F1": f1_v, "AUC": auc_v}

    p_t, r_t, f1_t = evaluate.precision_recall_f1(y_tier_test, preds_test_dt_clf)
    auc_t = evaluate.roc_auc(y_tier_test, scores_test_dt_clf)
    test_results["DecisionTree Clf (sklearn)"] = {"Precision": p_t, "Recall": r_t, "F1": f1_t, "AUC": auc_t}
    print(f"     Val F1: {f1_v:.4f}, AUC: {auc_v:.4f} (fit: {t_fit_dt_clf:.2f}s)")

    # -------------------------------------------------------------------------
    # 4. Error Analysis, Figure Generation & Table Output
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Generating error analysis figures and comparison reports...")

    if pegasos_svm.feature_map == "linear":
        plot_svm_weights(
            feat_names,
            pegasos_svm.w,
            FIG_DIR / "svm_feature_weights.png",
        )
    else:
        print("  - Skipped original-feature weight plot for RFF model")
    plot_tree_feature_importances(feat_names, dt_clf.feature_importances_, FIG_DIR / "tree_feature_importances.png")
    plot_error_analysis(
        y_val,
        preds_val_dt_reg,
        y_tier_val,
        preds_val_svm,
        FIG_DIR / "error_analysis.png",
    )
    print("  ✓ Saved figures to report/figures/:")
    print("     - svm_feature_weights.png")
    print("     - tree_feature_importances.png")
    print("     - error_analysis.png")

    svm_diagnostics = pd.DataFrame([
        {
            "model": f"PegasosSVM ({pegasos_svm.feature_map})",
            "feature_map": pegasos_svm.feature_map,
            "lambda": pegasos_svm.lambda_,
            "n_iters": pegasos_svm.n_iters,
            "gamma": pegasos_svm.gamma,
            "n_components": pegasos_svm.n_components,
            "weight_norm": np.linalg.norm(pegasos_svm.w),
            "margin_width": pegasos_svm.margin_width_,
            "support_vector_count": len(pegasos_svm.support_indices_),
            "final_objective": pegasos_svm.objective_history_[-1],
            "fit_seconds": t_fit_svm,
        },
        {
            "model": "PegasosSVM (RFF)",
            "feature_map": "rff",
            "lambda": pegasos_rff.lambda_,
            "n_iters": pegasos_rff.n_iters,
            "gamma": pegasos_rff.gamma,
            "n_components": pegasos_rff.n_components,
            "weight_norm": np.linalg.norm(pegasos_rff.w),
            "margin_width": pegasos_rff.margin_width_,
            "support_vector_count": len(pegasos_rff.support_indices_),
            "final_objective": pegasos_rff.objective_history_[-1],
            "fit_seconds": t_fit_rff,
        },
    ])

    svm_diagnostics.to_csv(
        REPORT_DIR / "svm_diagnostics.csv",
        index=False,
    )

    convergence = pd.DataFrame({
        "linear_iteration": pegasos_svm.objective_iterations_,
        "linear_objective": pegasos_svm.objective_history_,
        "rff_iteration": pegasos_rff.objective_iterations_,
        "rff_objective": pegasos_rff.objective_history_,
    })

    convergence.to_csv(
        REPORT_DIR / "svm_convergence.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("  VALIDATION SET BENCHMARK RESULTS")
    print("=" * 80)
    evaluate.compare(val_results)

    print("=" * 80)
    print("  TEST SET HEADLINE RESULTS (Unseen Evaluation)")
    print("=" * 80)
    evaluate.compare(test_results)

    # Save summary tables to CSV
    df_val = pd.DataFrame(val_results).T
    df_test = pd.DataFrame(test_results).T
    df_val.to_csv(REPORT_DIR / "validation_summary.csv")
    df_test.to_csv(REPORT_DIR / "test_summary.csv")
    print(
        f"  ✓ Exported results to {REPORT_DIR / 'validation_summary.csv'}, "
        f"{REPORT_DIR / 'test_summary.csv'}, "
        f"{REPORT_DIR / 'svm_validation_trials.csv'}, "
        f"{REPORT_DIR / 'svm_diagnostics.csv'}, and "
        f"{REPORT_DIR / 'svm_convergence.csv'}"
    )
    print("\n✓ Pipeline execution complete.")


if __name__ == "__main__":
    main()
