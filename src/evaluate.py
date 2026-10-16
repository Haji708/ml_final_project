"""
evaluate.py — metrics + comparison helpers (FROM SCRATCH where reasonable).

Implement the metrics yourself in NumPy. scikit-learn may be used ONLY to
sanity-check your numbers or for the baseline models you compare against.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Regression (Task A: price)
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error: sqrt( mean( (y_true - y_pred)^2 ) )."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error: mean( |y_true - y_pred| )."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    """
    Coefficient of determination R²:
        R² = 1 - SS_res / SS_tot
    where SS_res = sum((y_true - y_pred)^2)
      and SS_tot = sum((y_true - mean(y_true))^2).
    Returns 1.0 for a perfect model, ≤ 0 for a model worse than the mean.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        # All true values are identical — R² is undefined; return 1 if pred
        # also constant and equal, 0 otherwise (sklearn convention).
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Classification (Task B: price tier)
# ---------------------------------------------------------------------------

def confusion_matrix(y_true, y_pred):
    """
    Compute the 2×2 confusion matrix for binary labels.

    Layout (scikit-learn convention):
        [[TN, FP],
         [FN, TP]]

    Positive class is assumed to be 1 (premium).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    return np.array([[TN, FP],
                     [FN, TP]], dtype=int)


def precision_recall_f1(y_true, y_pred):
    """
    Return (precision, recall, f1) for the positive (premium, label=1) class.

    precision = TP / (TP + FP)   — of all predicted positives, how many are real?
    recall    = TP / (TP + FN)   — of all real positives, how many did we catch?
    f1        = 2 * P * R / (P + R)
    """
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return float(precision), float(recall), float(f1)


def roc_auc(y_true, scores) -> float:
    """
    Area Under the ROC Curve computed via the rank-based (Mann-Whitney U) method:

        AUC = (number of (pos, neg) pairs where score_pos > score_neg
               + 0.5 * ties) / (n_pos * n_neg)

    `scores` can be SVM decision-function values or tree class probabilities.
    """
    y_true  = np.asarray(y_true)
    scores  = np.asarray(scores, dtype=float)

    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc requires at least one positive and one negative sample.")

    # Vectorised Mann-Whitney: compare every positive score to every negative
    # shape (n_pos, n_neg)
    diff = pos_scores[:, None] - neg_scores[None, :]
    concordant = np.sum(diff > 0)
    ties       = np.sum(diff == 0)

    auc = (concordant + 0.5 * ties) / (n_pos * n_neg)
    return float(auc)


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare(name_to_metrics: dict) -> None:
    """
    Pretty-print a comparison table of models and their metrics.

    `name_to_metrics` maps a model name (str) to a dict of metric_name → value.
    Example:
        compare({
            "My Tree (reg)":   {"RMSE": 45000, "MAE": 30000, "R2": 0.78},
            "sklearn DT (reg)":{"RMSE": 44000, "MAE": 29500, "R2": 0.79},
            "My Tree (clf)":   {"Precision": 0.81, "Recall": 0.79, "F1": 0.80, "AUC": 0.87},
            "PegasosSVM":      {"Precision": 0.83, "Recall": 0.81, "F1": 0.82, "AUC": 0.89},
        })
    """
    if not name_to_metrics:
        print("(no models to compare)")
        return

    # Collect all metric names in insertion order
    all_metrics: list[str] = []
    for metrics in name_to_metrics.values():
        for k in metrics:
            if k not in all_metrics:
                all_metrics.append(k)

    # Column widths
    name_w   = max(len(n) for n in name_to_metrics) + 2
    metric_w = max(max(len(m) for m in all_metrics), 10) + 2

    # Header
    header = f"{'Model':<{name_w}}" + "".join(f"{m:>{metric_w}}" for m in all_metrics)
    sep    = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    for model_name, metrics in name_to_metrics.items():
        row = f"{model_name:<{name_w}}"
        for m in all_metrics:
            val = metrics.get(m, None)
            if val is None:
                cell = "—"
            elif isinstance(val, float):
                cell = f"{val:.4f}"
            else:
                cell = str(val)
            row += f"{cell:>{metric_w}}"
        print(row)

    print(sep + "\n")
