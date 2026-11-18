"""
Test-Time Entropy Minimization (TENT) adaptation logic.
Adapts affine parameters (BatchNorm / LayerNorm) using prediction Shannon entropy.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


def collect_norm_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Collects affine normalization parameters (weight and bias) from LayerNorm and BatchNorm layers.
    Freezes all other parameters.
    """
    params = []
    for name, param in model.named_parameters():
        if "norm" in name.lower() or "bn" in name.lower():
            param.requires_grad = True
            params.append(param)
        else:
            param.requires_grad = False
    return params


def tent_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes Shannon entropy of softmax predictions:
    H(p) = - \\sum_c p(c) \\log p(c)
    """
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
    return entropy
