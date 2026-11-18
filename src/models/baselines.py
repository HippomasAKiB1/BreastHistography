"""
Baselines for multi-magnification histopathology classification.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import timm

from .backbones import BACKBONES, BACKBONE_OUT_DIM, MAGNIFICATIONS


class ConcatFusionBaseline(nn.Module):
    """
    Standard early/late concatenation baseline without cross-attention or graph reasoning.
    Extracts features from all four magnifications, projects each to proj_dim,
    concatenates them, and feeds into a classification MLP.
    """
    def __init__(
        self,
        proj_dim: int = 256,
        num_binary: int = 2,
        pretrained: bool = False,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.branches = nn.ModuleDict({
            mag: timm.create_model(
                BACKBONES[mag],
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
            for mag in MAGNIFICATIONS
        })
        self.projs = nn.ModuleDict({
            mag: nn.Linear(BACKBONE_OUT_DIM[mag], proj_dim)
            for mag in MAGNIFICATIONS
        })
        self.classifier = nn.Sequential(
            nn.Linear(4 * proj_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_binary),
        )

    def forward(
        self,
        x40: torch.Tensor,
        x100: torch.Tensor,
        x200: torch.Tensor,
        x400: torch.Tensor,
    ) -> torch.Tensor:
        inputs = {"40X": x40, "100X": x100, "200X": x200, "400X": x400}
        feats = [self.projs[m](self.branches[m](inputs[m])) for m in MAGNIFICATIONS]
        concat_feats = torch.cat(feats, dim=1)
        return self.classifier(concat_feats)


class SingleMagBaseline(nn.Module):
    """
    Single-magnification baseline model.
    Processes one magnification scale using its assigned backbone and a linear classification head.
    """
    def __init__(
        self,
        mag: str = "200X",
        num_classes: int = 2,
        pretrained: bool = False,
    ):
        super().__init__()
        assert mag in MAGNIFICATIONS, f"Unknown magnification: {mag}. Expected one of {MAGNIFICATIONS}"
        self.mag = mag
        self.backbone_name = BACKBONES[mag]
        self.in_dim = BACKBONE_OUT_DIM[mag]
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.in_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.classifier(feat)
