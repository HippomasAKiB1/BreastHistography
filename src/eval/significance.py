"""
Statistical significance testing and confidence interval calculations.
Includes Edwards' continuity-corrected McNemar test and Student's t confidence intervals.
"""

from typing import Dict, Any, List, Union
import numpy as np
import scipy.stats as stats


def paired_mcnemar_test(
    y_true: Union[np.ndarray, list],
    y_pred_a: Union[np.ndarray, list],
    y_pred_b: Union[np.ndarray, list],
) -> Dict[str, Any]:
    """
    Computes Edwards' continuity-corrected McNemar test between models A and B.

    Null Hypothesis (H0): Both models have the same error rate (b == c).
    Test Statistic: \\chi^2 = (|b - c| - 1)^2 / (b + c) with df = 1.

    Args:
        y_true: Ground-truth labels.
        y_pred_a: Predictions from Model A.
        y_pred_b: Predictions from Model B.

    Returns:
        Dict with chi2, p_value, discordant counts b and c, and significance flag at alpha=0.05.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b: A correct, B incorrect
    # c: A incorrect, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    total_discordant = b + c
    if total_discordant == 0:
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "b": b,
            "c": c,
            "significant": False,
        }

    chi2 = (np.abs(b - c) - 1.0) ** 2 / total_discordant
    p_value = 1.0 - stats.chi2.cdf(chi2, df=1)

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "b": b,
        "c": c,
        "significant": bool(p_value < 0.05),
    }


def compute_t_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
) -> float:
    """
    Computes half-width of two-sided Student's t confidence interval:
    CI_half = t_{1-\\alpha/2, n-1} * (s / \\sqrt{n})

    Args:
        values: List of numeric values across cross-validation folds.
        confidence: Confidence level (default 0.95).

    Returns:
        Half-width margin of error (delta such that CI is mean +/- delta).
    """
    n = len(values)
    if n <= 1:
        return 0.0
    se = np.std(values, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1)
    return float(t_crit * se)
