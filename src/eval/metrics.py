"""
Scientifically rigorous evaluation metrics for histopathology classification.
Enforces mathematical sanity checks, confusion matrix invariants, calibration, and ECE.
"""

from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
    average_precision_score,
)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) for binary classification.

    Args:
        y_true: Binary ground-truth labels {0, 1}.
        y_prob: Predicted probabilities for positive class [0, 1].
        n_bins: Number of equal-width probability bins.

    Returns:
        Scalar ECE value in [0, 1].
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (n_in_bin / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_classification_metrics(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes a comprehensive suite of binary classification metrics.

    Args:
        y_true: 1D array of binary ground-truth labels {0, 1}.
        y_prob: 1D array of predicted probabilities for class 1, or 2D array [N, 2].
        threshold: Classification decision boundary threshold (default 0.5).

    Returns:
        Dictionary of computed metrics including accuracy, precision, recall, specificity,
        macro-F1, AUROC, AUPRC, Brier score, ECE, and confusion matrix counts.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_prob.ndim == 2:
        if y_prob.shape[1] == 2:
            y_prob_1 = y_prob[:, 1]
        else:
            raise ValueError(f"Expected 2 class columns, got shape {y_prob.shape}")
    else:
        y_prob_1 = y_prob

    y_pred = (y_prob_1 >= threshold).astype(int)
    n_samples = len(y_true)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    assert (tn + fp + fn + tp) == n_samples, f"Confusion matrix sum {tn + fp + fn + tp} != total {n_samples}"

    # Class prevalences
    prev_0 = float(np.mean(y_true == 0))
    prev_1 = float(np.mean(y_true == 1))

    # Mathematical sanity check on degenerate predictions
    pred_0_frac = np.mean(y_pred == 0)
    pred_1_frac = np.mean(y_pred == 1)
    acc = accuracy_score(y_true, y_pred)

    if pred_0_frac == 1.0:
        np.testing.assert_almost_equal(
            acc, prev_0, decimal=4,
            err_msg=f"Degenerate prediction (all 0) accuracy {acc:.4f} != class 0 prevalence {prev_0:.4f}"
        )
    elif pred_1_frac == 1.0:
        np.testing.assert_almost_equal(
            acc, prev_1, decimal=4,
            err_msg=f"Degenerate prediction (all 1) accuracy {acc:.4f} != class 1 prevalence {prev_1:.4f}"
        )

    # Standard metrics
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_binary = f1_score(y_true, y_pred, average="binary", zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # AUROC & AUPRC
    try:
        auroc = float(roc_auc_score(y_true, y_prob_1))
    except ValueError:
        auroc = None

    try:
        auprc = float(average_precision_score(y_true, y_prob_1))
    except ValueError:
        auprc = None

    brier = float(brier_score_loss(y_true, y_prob_1))
    ece = compute_ece(y_true, y_prob_1, n_bins=10)

    return {
        "n_samples": int(n_samples),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "npv": float(npv),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_binary": float(f1_binary),
        "auroc": auroc,
        "auprc": auprc,
        "brier_score": brier,
        "ece": ece,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "prevalence_class_0": prev_0,
        "prevalence_class_1": prev_1,
    }
