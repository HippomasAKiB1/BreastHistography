"""
Unit tests and mathematical sanity proofs for evaluation metrics and statistical tests.
"""

import pytest
import numpy as np

from src.eval import (
    compute_classification_metrics,
    compute_ece,
    paired_mcnemar_test,
    compute_t_confidence_interval,
)


def test_metric_invariants_and_collapse_bounds():
    # Construct exact test distribution: 31 benign (0), 69 malignant (1)
    y_true = np.array([0] * 31 + [1] * 69)

    # All-0 prediction collapse
    y_pred_0 = np.zeros(100)
    m0 = compute_classification_metrics(y_true, y_pred_0)
    assert abs(m0["accuracy"] - 0.31) < 1e-4, f"All-0 accuracy {m0['accuracy']} != class 0 prevalence 0.31"
    assert m0["recall"] == 0.0
    assert m0["specificity"] == 1.0

    # All-1 prediction collapse
    y_pred_1 = np.ones(100)
    m1 = compute_classification_metrics(y_true, y_pred_1)
    assert abs(m1["accuracy"] - 0.69) < 1e-4, f"All-1 accuracy {m1['accuracy']} != class 1 prevalence 0.69"
    assert m1["recall"] == 1.0
    assert m1["specificity"] == 0.0

    # Perfect prediction
    y_perfect = y_true.astype(float)
    mp = compute_classification_metrics(y_true, y_perfect)
    assert mp["accuracy"] == 1.0
    assert mp["f1_macro"] == 1.0
    assert mp["auroc"] == 1.0
    assert mp["ece"] < 1e-4


def test_ece_computation():
    y_true = np.array([0, 0, 1, 1])
    # Perfectly calibrated probabilities
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    ece = compute_ece(y_true, y_prob, n_bins=5)
    assert ece == 0.0, f"Expected 0.0 ECE for perfect calibration, got {ece}"

    # Overconfident wrong predictions
    y_prob_wrong = np.array([1.0, 1.0, 0.0, 0.0])
    ece_wrong = compute_ece(y_true, y_prob_wrong, n_bins=5)
    assert ece_wrong > 0.8, f"Expected high ECE for wrong confident predictions, got {ece_wrong}"


def test_mcnemar_significance():
    y_true = np.array([0] * 10 + [1] * 10)
    # Model A: 18 correct, 2 incorrect
    pred_a = np.array([0] * 9 + [1] + [1] * 9 + [0])
    # Model B: 10 correct, 10 incorrect
    pred_b = np.array([1] * 10 + [0] * 10)

    res = paired_mcnemar_test(y_true, pred_a, pred_b)
    assert "p_value" in res
    assert "chi2" in res
    assert res["b"] >= res["c"]
    assert res["significant"] is True


def test_t_confidence_interval():
    folds = [95.0, 96.0, 97.0, 95.5, 96.5]
    ci = compute_t_confidence_interval(folds, confidence=0.95)
    assert ci > 0.0
    assert ci < 5.0, f"Confidence interval half-width seems too wide: {ci}"
