"""
Fast Test-Time Adaptation (TTA) MIL Adapter.
Performs test-time optimization per patient slide/bag and microsecond state restoration.
"""

from typing import Optional, Dict, Any, List
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tent import collect_norm_parameters, tent_loss
from .hat_mil import collect_hat_parameters, prototype_regularized_loss


class FastTTAMILAdapter:
    """
    Online test-time adaptation wrapper.
    Adapts test samples without ground-truth labels and restores initial parameters.
    """
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        source_cov: Optional[torch.Tensor] = None,
        source_prototypes: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.lr = lr
        self.source_cov = source_cov
        self.source_prototypes = source_prototypes
        # Cache initial state for microsecond restoration
        self.init_state = {k: v.clone() for k, v in model.state_dict().items()}

    def reset(self) -> None:
        """Restores model to its original un-adapted checkpoint state."""
        self.model.load_state_dict(self.init_state)

    def adapt_bag(
        self,
        feat: torch.Tensor,
        method: str = "no_tta",
        steps: int = 3,
    ) -> torch.Tensor:
        """
        Adapts model parameters on an unlabelled bag feature tensor [1, N, D].
        Resets weights immediately after prediction to prevent cumulative drift.

        Args:
            feat: Feature tensor of shape [1, N, D] or [B, D].
            method: Adaptation strategy: 'no_tta', 'tent_1', 'tent_10', 'coral_5', 'hat_mil'.
            steps: Number of gradient update steps.

        Returns:
            Adapted logits tensor.
        """
        if method == "no_tta":
            with torch.no_grad():
                out = self.model(feat)
                logits = out[0] if isinstance(out, (tuple, list)) else out
            return logits

        self.reset()
        self.model.train()

        if method in ["tent_1", "tent_10"]:
            max_steps = 1 if method == "tent_1" else steps
            params = collect_norm_parameters(self.model)
            optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9)

            for _ in range(max_steps):
                optimizer.zero_grad()
                out = self.model(feat)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                loss = tent_loss(logits)
                loss.backward()
                optimizer.step()

        elif method == "coral_5":
            params = collect_norm_parameters(self.model)
            optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9)

            for _ in range(steps):
                optimizer.zero_grad()
                out = self.model(feat)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()

                patches = feat.squeeze(0)
                n, d = patches.shape
                patches_norm = self.model.norm(patches) if hasattr(self.model, "norm") else patches
                mean_t = patches_norm.mean(dim=0, keepdim=True)
                cov_t = (patches_norm - mean_t).T @ (patches_norm - mean_t) / (n - 1 + 1e-8)

                if self.source_cov is not None:
                    coral_loss = torch.norm(self.source_cov.to(cov_t.device) - cov_t, p="fro") / (4.0 * d * d)
                else:
                    coral_loss = torch.tensor(0.0, device=cov_t.device)

                loss = entropy + 0.5 * coral_loss
                loss.backward()
                optimizer.step()

        elif method == "hat_mil":
            params = collect_hat_parameters(self.model)
            optimizer = torch.optim.SGD(params, lr=self.lr * 0.5, momentum=0.9)

            for _ in range(steps):
                optimizer.zero_grad()
                out = self.model(feat)
                logits = out[0]
                z = out[1] if len(out) > 1 else feat.mean(dim=1)

                if self.source_prototypes is not None:
                    loss, _, _ = prototype_regularized_loss(
                        logits=logits,
                        features=z,
                        prototypes=self.source_prototypes,
                        proto_weight=0.4,
                    )
                else:
                    loss = tent_loss(logits)

                loss.backward()
                optimizer.step()

        with torch.no_grad():
            self.model.eval()
            out = self.model(feat)
            logits = out[0] if isinstance(out, (tuple, list)) else out

        self.reset()
        return logits
