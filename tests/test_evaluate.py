"""
Unit tests for src/evaluate.py:
Verifies that from-scratch metrics match scikit-learn reference values exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as sk_metrics

from src import evaluate


@pytest.fixture
def regression_data():
    np.random.seed(42)
    y_true = np.array([120000.0, 250000.0, 85000.0, 430000.0, 195000.0, 310000.0])
    y_pred = np.array([115000.0, 270000.0, 92000.0, 410000.0, 190000.0, 325000.0])
    return y_true, y_pred


@pytest.fixture
def classification_data():
    np.random.seed(42)
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    scores = np.array([0.9, 0.1, 0.85, 0.45, 0.2, 0.65, 0.78, 0.15, 0.92, 0.3])
    return y_true, y_pred, scores


def test_rmse(regression_data):
    y_true, y_pred = regression_data
    val = evaluate.rmse(y_true, y_pred)
    sk_val = float(np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred)))
    assert np.isclose(val, sk_val, rtol=1e-5)


def test_mae(regression_data):
    y_true, y_pred = regression_data
    val = evaluate.mae(y_true, y_pred)
    sk_val = float(sk_metrics.mean_absolute_error(y_true, y_pred))
    assert np.isclose(val, sk_val, rtol=1e-5)


def test_r2(regression_data):
    y_true, y_pred = regression_data
    val = evaluate.r2(y_true, y_pred)
    sk_val = float(sk_metrics.r2_score(y_true, y_pred))
    assert np.isclose(val, sk_val, rtol=1e-5)


def test_confusion_matrix(classification_data):
    y_true, y_pred, _ = classification_data
    cm = evaluate.confusion_matrix(y_true, y_pred)
    sk_cm = sk_metrics.confusion_matrix(y_true, y_pred)
    assert np.array_equal(cm, sk_cm)


def test_precision_recall_f1(classification_data):
    y_true, y_pred, _ = classification_data
    p, r, f1 = evaluate.precision_recall_f1(y_true, y_pred)
    sk_p = float(sk_metrics.precision_score(y_true, y_pred, pos_label=1))
    sk_r = float(sk_metrics.recall_score(y_true, y_pred, pos_label=1))
    sk_f1 = float(sk_metrics.f1_score(y_true, y_pred, pos_label=1))
    assert np.isclose(p, sk_p, rtol=1e-5)
    assert np.isclose(r, sk_r, rtol=1e-5)
    assert np.isclose(f1, sk_f1, rtol=1e-5)


def test_roc_auc(classification_data):
    y_true, _, scores = classification_data
    auc = evaluate.roc_auc(y_true, scores)
    sk_auc = float(sk_metrics.roc_auc_score(y_true, scores))
    assert np.isclose(auc, sk_auc, rtol=1e-5)


def test_compare_output(capsys):
    results = {
        "Model A": {"RMSE": 1000.0, "R2": 0.85},
        "Model B": {"Precision": 0.90, "Recall": 0.88, "F1": 0.89},
    }
    evaluate.compare(results)
    captured = capsys.readouterr()
    assert "Model A" in captured.out
    assert "Model B" in captured.out
    assert "0.8500" in captured.out
    assert "—" in captured.out
