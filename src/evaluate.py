"""
Metrics implemented with NumPy and a model-comparison table.
"""

from __future__ import annotations

import numpy as np


def _matching_arrays(y_true, y_pred):
    """Return validated one-dimensional arrays."""
    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    for name, array in [("y_true", true), ("y_pred", predicted)]:
        if array.ndim != 1 and not (array.ndim == 2 and array.shape[1] == 1):
            raise ValueError(f"{name} must be 1D or an (n,1) column")
    true, predicted = true.reshape(-1), predicted.reshape(-1)

    if len(true) == 0:
        raise ValueError("Arrays cannot be empty")

    if true.shape != predicted.shape:
        raise ValueError("Arrays must have the same shape")
    if not np.isfinite(true.astype(float)).all() or not np.isfinite(predicted.astype(float)).all():
        raise ValueError("Arrays must contain finite numbers")

    return true, predicted


# --- Regression -------------------------------------------------------------

def rmse(y_true, y_pred) -> float:
    """Return root mean squared error."""
    true, predicted = _matching_arrays(y_true, y_pred)
    true = true.astype(float)
    predicted = predicted.astype(float)

    return float(
        np.sqrt(np.mean((true - predicted) ** 2))
    )


def mae(y_true, y_pred) -> float:
    """Return mean absolute error."""
    true, predicted = _matching_arrays(y_true, y_pred)
    true = true.astype(float)
    predicted = predicted.astype(float)

    return float(np.mean(np.abs(true - predicted)))


def r2(y_true, y_pred) -> float:
    """Return the coefficient of determination."""
    true, predicted = _matching_arrays(y_true, y_pred)
    true = true.astype(float)
    predicted = predicted.astype(float)

    residual_sum = np.sum((true - predicted) ** 2)
    if len(true) < 2:
        return float("nan")
    total_sum = np.sum((true - np.mean(true)) ** 2)

    if total_sum == 0:
        return 1.0 if residual_sum == 0 else 0.0

    return float(1.0 - residual_sum / total_sum)


# --- Classification ---------------------------------------------------------

def confusion_matrix(y_true, y_pred):
    """Return the matrix [[TN, FP], [FN, TP]]."""
    true, predicted = _matching_arrays(y_true, y_pred)
    if not np.isin(true, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Binary classification labels must be 0 or 1")
    true = true.astype(int)
    predicted = predicted.astype(int)

    true_negative = np.sum((true == 0) & (predicted == 0))
    false_positive = np.sum((true == 0) & (predicted == 1))
    false_negative = np.sum((true == 1) & (predicted == 0))
    true_positive = np.sum((true == 1) & (predicted == 1))

    return np.array(
        [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        dtype=int,
    )


def precision_recall_f1(y_true, y_pred):
    """Return precision, recall, and F1 for class 1."""
    matrix = confusion_matrix(y_true, y_pred)

    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )
    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return float(precision), float(recall), float(f1)


def roc_auc(y_true, scores) -> float:
    """
    Return rank-based ROC AUC in O(n log n); tied scores receive average ranks.
    """
    true, score_array = _matching_arrays(y_true, scores)
    if not np.isin(true, [0, 1]).all():
        raise ValueError("ROC AUC labels must be 0 or 1")
    true = true.astype(int)
    score_array = score_array.astype(float)

    if true.shape != score_array.shape:
        raise ValueError("y_true and scores must have the same shape")

    positive_scores = score_array[true == 1]
    negative_scores = score_array[true == 0]

    if len(positive_scores) == 0 or len(negative_scores) == 0:
        raise ValueError("ROC AUC requires both classes")

    order = np.argsort(score_array, kind="mergesort")
    sorted_scores = score_array[order]
    ranks = np.empty(len(score_array), dtype=float)

    start = 0

    while start < len(sorted_scores):
        end = start + 1

        while (
            end < len(sorted_scores)
            and sorted_scores[end] == sorted_scores[start]
        ):
            end += 1

        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = np.sum(ranks[true == 1])
    positive_count = len(positive_scores)
    negative_count = len(negative_scores)

    auc = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)

    return float(auc)


# --- Comparison helper ------------------------------------------------------

def compare(name_to_metrics: dict) -> None:
    """Pretty-print model metrics as a comparison table."""
    if not name_to_metrics:
        print("No model results were supplied.")
        return

    metric_names = sorted(
        {
            metric_name
            for metrics in name_to_metrics.values()
            for metric_name in metrics
        }
    )

    model_width = max(
        len("Model"),
        max(len(str(name)) for name in name_to_metrics),
    )
    metric_width = 14

    header = (
        f"{'Model':<{model_width}}"
        + "".join(
            f"{metric:>{metric_width}}"
            for metric in metric_names
        )
    )

    print(header)
    print("-" * len(header))

    for model_name, metrics in name_to_metrics.items():
        row = f"{model_name:<{model_width}}"

        for metric_name in metric_names:
            value = metrics.get(metric_name)

            if value is None:
                row += f"{'-':>{metric_width}}"
            else:
                row += f"{float(value):>{metric_width}.4f}"

        print(row)
