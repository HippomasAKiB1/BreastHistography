"""
Evaluation metrics, calibration error, and statistical significance testing.
"""

from .metrics import compute_classification_metrics, compute_ece
from .significance import paired_mcnemar_test, compute_t_confidence_interval

__all__ = [
    "compute_classification_metrics",
    "compute_ece",
    "paired_mcnemar_test",
    "compute_t_confidence_interval",
]
