"""
Head-Aware Test-Time Adaptation (HAT-MIL) with Prototype Regularization.
Addresses single-class collapse in multi-instance histopathology bags
by jointly adapting normalization parameters and classifier head with class prototype alignment.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def collect_hat_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Collects parameters for Head-Aware Adaptation:
    Normalization layers + Classifier projection weights/biases.
    """
    params = []
    for name, param in model.named_parameters():
        if "norm" in name.lower() or "classifier" in name.lower() or "head" in name.lower():
            param.requires_grad = True
            params.append(param)
        else:
            param.requires_grad = False
    return params


def prototype_regularized_loss(
    logits: torch.Tensor,
    features: torch.Tensor,
    prototypes: torch.Tensor,
    proto_weight: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes joint Shannon entropy and prototype MSE loss:
    L = H(p) + \\lambda_{proto} * ||z - p_{y}||^2_2

    Args:
        logits: Bag or instance classification logits [B, C].
        features: Bag-level representation vectors [B, D].
        prototypes: Source training prototypes [C, D].
        proto_weight: Regularization coefficient lambda (default 0.4).

    Returns:
        Tuple of (total_loss, entropy_loss, proto_loss).
    """
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()

    # Pseudo-label from high-confidence predictions
    pseudo_labels = torch.argmax(probs, dim=-1)
    selected_prototypes = prototypes[pseudo_labels].to(features.device)
    proto_loss = F.mse_loss(features, selected_prototypes)

    total_loss = entropy + proto_weight * proto_loss
    return total_loss, entropy, proto_loss
