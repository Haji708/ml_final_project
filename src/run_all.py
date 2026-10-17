"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all

This orchestrator:
  1. Loads and cleans the bina.az dataset with training-fitted preprocessing,
  2. Builds features, stratified splits, and standardized matrices,
  3. Selects Linear/Ridge features, targets and alpha on validation; trains a tree baseline,
  4. Trains Task B classification models (from-scratch PegasosSVM, sklearn SGDClassifier, LinearSVC, DecisionTreeClassifier),
  5. Evaluates all models using our from-scratch src.evaluate suite,
  6. Generates error analysis and model interpretation figures into report/figures/,
  7. Saves validation/test tables, selected settings, and validation error diagnostics.

Randomized operations use the requested seed (default 42).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Support both `python -m src.run_all` and `python src/run_all.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_prep, evaluate, member1_regression
from src.svm import PegasosSVM

# Optional from-scratch decision tree
try:
    from src.decision_tree import DecisionTree
except ImportError:
    DecisionTree = None

from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

REPORT_DIR = data_prep.PROJECT_ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"


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
    plt.title("Top 15 Feature Importances — scratch DecisionTree", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tree_validation_curves(
        regression_trials: pd.DataFrame,
        classification_trials: pd.DataFrame,
        save_path: Path,
) -> None:
    """Plot scratch-tree training and validation scores across depth settings."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for leaf_size, trials in regression_trials.groupby("min_samples_leaf"):
        trials = trials.sort_values("max_depth")
        axes[0].plot(
            trials["max_depth"], trials["training_mae"],
            marker="o", linestyle="--", label=f"leaf={leaf_size} train",
        )
        axes[0].plot(
            trials["max_depth"], trials["validation_mae"],
            marker="o", label=f"leaf={leaf_size} validation",
        )
    axes[0].set_title("Scratch Regression Tree")
    axes[0].set_ylabel("MAE (AZN)")
    axes[0].set_xlabel("Maximum depth")
    axes[0].legend(fontsize=8)

    for axis, criterion in zip(axes[1:], ("gini", "entropy")):
        criterion_trials = classification_trials[
            classification_trials["criterion"] == criterion
            ]
        for leaf_size, trials in criterion_trials.groupby("min_samples_leaf"):
            trials = trials.sort_values("max_depth")
            axis.plot(
                trials["max_depth"], trials["training_f1"],
                marker="o", linestyle="--", label=f"leaf={leaf_size} train",
            )
            axis.plot(
                trials["max_depth"], trials["validation_f1"],
                marker="o", label=f"leaf={leaf_size} validation",
            )
        axis.set_title(f"Scratch Classification Tree ({criterion})")
        axis.set_xlabel("Maximum depth")
        axis.legend(fontsize=8)

    for axis in axes:
        axis.set_xticks(sorted(regression_trials["max_depth"].unique()))
        axis.grid(True, linestyle=":", alpha=0.6)
    axes[1].set_ylabel("F1 score")
    axes[2].set_ylabel("F1 score")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


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
        X_train, y_train, X_validation, y_validation, *,
        iteration_budgets=(50_000, 200_000, 500_000),
        component_counts=(250, 500),
        gammas=(0.001, 0.005, 0.01, 0.05, 0.5, 1.0, 2.0),
        random_state=42, verbose=False,
):
    """Coarse search, then refine each family's winner using validation only.

    Budgets count single-example updates, not epochs. Keep the short-budget
    candidates in the final comparison so longer training must earn its place.
    RFF width is expanded only after choosing a promising gamma/lambda pair.
    """
    budgets = sorted(set(int(value) for value in iteration_budgets))
    components = sorted(set(int(value) for value in component_counts))
    if not budgets or budgets[0] <= 0 or not components or components[0] <= 0:
        raise ValueError("SVM iteration budgets and component counts must be positive")
    if not gammas or any(not np.isfinite(value) or value <= 0 for value in gammas):
        raise ValueError("SVM gammas must be finite and positive")
    trials, seen = [], set()

    def train_candidate(config, phase):
        key = tuple(config[name] for name in
                    ("feature_map", "lambda_", "gamma", "n_components", "n_iters"))
        if key in seen:
            return
        seen.add(key)
        started = time.perf_counter()
        model = PegasosSVM(**config, random_state=random_state,
                           record_every=max(1, config["n_iters"] // 10))
        model.fit(X_train, y_train)
        prediction = model.predict(X_validation)
        scores = model.decision_function(X_validation)
        precision, recall, f1 = evaluate.precision_recall_f1(y_validation, prediction)
        trials.append({**config, "phase": phase,
                       "Precision": precision, "Recall": recall, "F1": f1,
                       "AUC": evaluate.roc_auc(y_validation, scores),
                       "updates_per_training_row": config["n_iters"] / len(X_train),
                       "initial_objective": model.objective_history_[0],
                       "final_objective": model.objective_history_[-1],
                       "fit_and_validation_seconds": time.perf_counter() - started})
        if verbose:
            print(f"     {config['feature_map']}: lambda={config['lambda_']:g}, "
                  f"gamma={config['gamma']:g}, components={config['n_components']}, "
                  f"updates={config['n_iters']:,}, validation F1={f1:.4f}", flush=True)

    for regularization in (1e-5, 1e-4, 1e-3):
        train_candidate({"feature_map": "linear", "lambda_": regularization,
                         "gamma": 1.0, "n_components": components[0],
                         "n_iters": budgets[0]}, "coarse")
    for gamma in gammas:
        for regularization in (1e-4, 1e-3):
            train_candidate({"feature_map": "rff", "lambda_": regularization,
                             "gamma": float(gamma), "n_components": components[0],
                             "n_iters": budgets[0]}, "coarse")

    for feature_map in ("linear", "rff"):
        table = pd.DataFrame(trials)
        winner = table[table["feature_map"] == feature_map].sort_values(
            ["F1", "AUC"], ascending=False, kind="stable").iloc[0]
        widths = components if feature_map == "rff" else components[:1]
        for width in widths:
            for budget in budgets:
                train_candidate({"feature_map": feature_map,
                                 "lambda_": float(winner["lambda_"]),
                                 "gamma": float(winner["gamma"]),
                                 "n_components": width, "n_iters": budget}, "refine")
    trials_df = pd.DataFrame(trials)
    best_config = trials_df.sort_values(
        ["F1", "AUC"], ascending=False, kind="stable").iloc[0].to_dict()
    return best_config, trials_df


def svm_from_trial(trial, seed):
    """Reproduce the exact validation-selected training configuration."""
    updates = int(trial["n_iters"])
    return PegasosSVM(lambda_=float(trial["lambda_"]), n_iters=updates,
                      random_state=seed, feature_map=str(trial["feature_map"]),
                      n_components=int(trial["n_components"]), gamma=float(trial["gamma"]),
                      record_every=max(1, updates // 10))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the shared validation-selected benchmark")
    parser.add_argument("--data", default=None, help="Path to bina_az_sale.csv")
    parser.add_argument("--output", type=Path, default=data_prep.PROJECT_ROOT / "report")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=data_prep.SEED)
    parser.add_argument("--svm-iterations", type=int, nargs="+", default=[50_000, 200_000, 500_000],
                        help="Candidate numbers of single-example SVM updates")
    parser.add_argument("--svm-components", type=int, nargs="+", default=[250, 500])
    args = parser.parse_args(argv)
    global REPORT_DIR, FIG_DIR
    REPORT_DIR = args.output.resolve()
    FIG_DIR = REPORT_DIR / "figures"
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
    bundle = data_prep.prepare_data(args.data, seed=args.seed)
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
    scratch_tree_configs = {}

    # -------------------------------------------------------------------------
    # 2. Task A: Regression (Price Prediction)
    # -------------------------------------------------------------------------
    print("\n[Step 2/4] Training and evaluating Task A regression models...")

    regression = member1_regression.run_member1_experiments(
        bundle, REPORT_DIR / "member1_checked", cv_folds=args.cv_folds,
        include_tree_baseline=True,
    )
    regression_names = {
        "Custom Linear": "Linear (Scratch)", "sklearn Linear": "Linear (sklearn)",
        "Custom Ridge": "Ridge (Scratch)", "sklearn Ridge": "Ridge (sklearn)",
        "sklearn DecisionTreeRegressor": "DecisionTree Reg (sklearn)",
        "Training median": "Training median",
    }
    for source_name, display_name in regression_names.items():
        val_results[display_name] = regression["validation_results"][source_name]
        test_results[display_name] = regression["regression_results"][source_name]
    preds_val_dt_reg = regression["validation_predictions"]["sklearn DecisionTreeRegressor"]

    # Model A4: From-scratch DecisionTree (if implemented)
    if DecisionTree is not None:
        t_start_tree = time.time()
        print("Tuning Scratch Regression Decision Tree ")
        # candidate_depths = [4, 6, 10, 12,16]
        # candidate_leaf_sizes = [1, 5, 10, 20,30]
        # candidate_split_sizes = [2, 10, 12,15,20]
        candidate_depths = [5, 10, 15]
        candidate_leaf_sizes = [1, 5, 10]

        best_val_score = float("inf")
        best_depth, best_leaf_size = (0, 0)
        best_tree = None
        test_result_scratch_des_tree = {}
        val_result_scratch_des_tree = {}
        regression_tree_trials = []
        for candidate_depth in candidate_depths:
            for candidate_leaf_size in candidate_leaf_sizes:

                tree_reg = DecisionTree(task="regression",
                                        criterion="mse", max_depth=candidate_depth,
                                        random_state=args.seed,
                                        min_samples_leaf=candidate_leaf_size,

                                        )
                tree_reg.fit(X_train, y_train)
                p_train = tree_reg.predict(X_train)
                p_val = tree_reg.predict(X_val)
                mae = evaluate.mae(y_val, p_val)
                regression_tree_trials.append({
                    "max_depth": candidate_depth,
                    "min_samples_leaf": candidate_leaf_size,
                    "training_mae": evaluate.mae(y_train, p_train),
                    "validation_mae": mae,
                })

                if mae < best_val_score:
                    best_val_score = mae
                    best_depth, best_leaf_size = candidate_depth, candidate_leaf_size
                    best_tree = tree_reg
                    val_result_scratch_des_tree = {
                        "RMSE": evaluate.rmse(y_val, p_val),
                        "MAE": evaluate.mae(y_val, p_val),
                        "R2": evaluate.r2(y_val, p_val),
                    }

        if best_tree is None:
            raise RuntimeError("No decision-tree configuration was selected")
        scratch_tree_configs["scratch_tree_regression"] = {
            "task": "regression",
            "criterion": "mse",
            "max_depth": best_depth,
            "min_samples_leaf": best_leaf_size,
            "random_state": args.seed,
        }
        t_end_tree = time.time()
        print(f"Tuning Regression Tree took {t_end_tree - t_start_tree:.6f} seconds")

        val_results["DecisionTree Reg (Scratch)"] = val_result_scratch_des_tree
        print(
            f"min_samples_leaf={best_leaf_size}, "
            f"validation MAE={best_val_score:.2f}"
        )

        p_test = best_tree.predict(X_test)

        test_result_scratch_des_tree = {
            "RMSE": evaluate.rmse(y_test, p_test),
            "MAE": evaluate.mae(y_test, p_test),
            "R2": evaluate.r2(y_test, p_test),
        }

        test_results["DecisionTree Reg (Scratch)"] = test_result_scratch_des_tree
        best_tree.compute_feature_importances(X_train,y_train)
        plot_tree_feature_importances(
            feat_names,
            best_tree.feature_importances_,
            FIG_DIR / "tree_regression_feature_importances.png",
        )
    else:
        regression_tree_trials = pd.DataFrame()

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
        iteration_budgets=args.svm_iterations, component_counts=args.svm_components,
        random_state=args.seed, verbose=True,
    )
    selected_linear = svm_trials[svm_trials["feature_map"] == "linear"].sort_values(
        ["F1", "AUC"], ascending=False, kind="stable").iloc[0].to_dict()
    selected_rff = svm_trials[svm_trials["feature_map"] == "rff"].sort_values(
        ["F1", "AUC"], ascending=False, kind="stable").iloc[0].to_dict()

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
    # Model B0: Decision Tree with tuned HyperParams
    if DecisionTree is not None:
        print("Tuning Scratch Decision Tree Classifier")
        t_start_tree = time.time()
        # candidate_depths = [4, 6, 10, 12,16]
        # candidate_leaf_sizes = [1, 5, 10, 20,30]
        # candidate_split_sizes = [2, 10, 12,15,20]
        candidate_depths = [5, 10, 15]
        candidate_leaf_sizes = [1, 5, 10]

        best_val_score = float("-inf")
        best_depth, best_leaf_size = (0, 0)
        best_tree = None
        test_result_scratch_tree = {}
        val_result_scratch_des_tree = {}
        classification_tree_trials = []
        for criterion in ["gini", 'entropy']:
            for candidate_depth in candidate_depths:
                for candidate_leaf_size in candidate_leaf_sizes:

                    tree_clf = DecisionTree(task="classification",
                                            criterion=criterion, max_depth=candidate_depth,
                                            random_state=args.seed,
                                            min_samples_leaf=candidate_leaf_size,

                                            )
                    tree_clf.fit(X_train, y_tier_tr)
                    p_train = tree_clf.predict(X_train)
                    p_val = tree_clf.predict(X_val)
                    scores_val = tree_clf.predict_proba(X_val)[:, 1]

                    _, _, training_f1 = evaluate.precision_recall_f1(
                        y_tier_tr,
                        p_train,
                    )
                    precision, recall, f1 = evaluate.precision_recall_f1(y_tier_val, p_val)
                    auc = evaluate.roc_auc(y_tier_val, scores_val)
                    classification_tree_trials.append({
                        "criterion": criterion,
                        "max_depth": candidate_depth,
                        "min_samples_leaf": candidate_leaf_size,
                        "training_f1": training_f1,
                        "validation_f1": f1,
                    })

                    if f1 > best_val_score:
                        best_val_score = f1
                        best_depth = candidate_depth
                        best_leaf_size = candidate_leaf_size
                        best_criterion = criterion
                        best_tree = tree_clf
                        val_result_scratch_tree = {
                            "Precision": precision,
                            "Recall": recall,
                            "F1": f1,
                            "AUC": auc,
                        }

        if best_tree is None:
            raise RuntimeError("No decision-tree configuration was selected")
        scratch_tree_configs["scratch_tree_classification"] = {
            "task": "classification",
            "criterion": best_criterion,
            "max_depth": best_depth,
            "min_samples_leaf": best_leaf_size,
            "random_state": args.seed,
        }
        t_end_tree = time.time()
        print(f"Tuning Classification Tree took {t_end_tree - t_start_tree:.6f} seconds")
        val_results["DecisionTree Classifier (Scratch)"] = val_result_scratch_tree
        print(
            f"Selected scratch tree: criterion={best_criterion}, depth={best_depth}, "
            f"min_samples_leaf={best_leaf_size}, "
            f"validation F1={best_val_score:.2f}"
        )

        scores_test = best_tree.predict_proba(X_test)[:, 1]
        p_test = best_tree.predict(X_test)

        precision, recall, f1 = evaluate.precision_recall_f1(y_tier_test, p_test)
        auc = evaluate.roc_auc(y_tier_test, scores_test)

        test_result_scratch_tree = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUC": auc,
        }

        test_results["DecisionTree Classifier (Scratch)"] = test_result_scratch_tree
        best_tree.compute_feature_importances(X_train,y_tier_tr)
        plot_tree_feature_importances(
            feat_names,
            best_tree.feature_importances_,
            FIG_DIR / "tree_classification_feature_importances.png",
        )
        
    else:
        classification_tree_trials = pd.DataFrame()

    # Report separately tuned linear and nonlinear SVMs under clear labels.
    print("  -> Training validation-selected linear Pegasos SVM...")
    t_start = time.time()
    pegasos_svm = svm_from_trial(selected_linear, args.seed)
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

    pegasos_rff = svm_from_trial(selected_rff, args.seed)
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
    sgd_clf = SGDClassifier(loss="hinge", alpha=1e-4, random_state=args.seed)
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
    lsvc = LinearSVC(C=1.0, random_state=args.seed, max_iter=2000)
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
    dt_clf = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=args.seed)
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

    regression_tree_trials = pd.DataFrame(regression_tree_trials)
    classification_tree_trials = pd.DataFrame(classification_tree_trials)
    if not regression_tree_trials.empty and not classification_tree_trials.empty:
        plot_tree_validation_curves(
            regression_tree_trials,
            classification_tree_trials,
            FIG_DIR / "tree_validation_curves.png",
        )

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
    if pegasos_svm.feature_map == "linear":
        print("     - svm_feature_weights.png")
    print("     - tree_feature_importances.png")
    print("     - tree_validation_curves.png")
    print("     - error_analysis.png")

    selections = {"seed": args.seed, "regression": regression["selections"],
                  "svm_linear": selected_linear, "svm_rff": selected_rff,
                  **scratch_tree_configs,
                  "selection_data": "training and validation only"}
    (REPORT_DIR / "selected_configs.json").write_text(
        json.dumps(selections, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

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
        f"{REPORT_DIR / 'test_summary.csv'}, and "
        f"{REPORT_DIR / 'selected_configs.json'}"
    )
    print("\n✓ Pipeline execution complete.")


if __name__ == "__main__":
    main()
