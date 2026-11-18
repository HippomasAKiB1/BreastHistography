"""
Test-Time Adaptation (TTA) algorithms and adapters for BreastHistography.
"""

from .tent import collect_norm_parameters, tent_loss
from .hat_mil import collect_hat_parameters, prototype_regularized_loss
from .adapter import FastTTAMILAdapter

__all__ = [
    "collect_norm_parameters",
    "tent_loss",
    "collect_hat_parameters",
    "prototype_regularized_loss",
    "FastTTAMILAdapter",
]
