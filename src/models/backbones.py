from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import timm

MAGNIFICATIONS: List[str] = ["40X", "100X", "200X", "400X"]

BACKBONES: Dict[str, str] = {
    "40X": "mobilenetv3_large_100",
    "100X": "efficientnet_b0",
    "200X": "resnet34",
    "400X": "mobilenetv3_small_100",
}

BACKBONE_OUT_DIM: Dict[str, int] = {
    "40X": 1280,
    "100X": 1280,
    "200X": 512,
    "400X": 1024,
}


class BranchExtractor(nn.Module):
    """
    Single pretrained vision backbone with linear feature projection
    and LayerNorm for a specific optical magnification scale.
    """
    def __init__(self, backbone_name: str, in_dim: int, out_dim: int = 512, pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self._raw_dim = in_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.backbone(x)  # [B, in_dim]
        proj = self.proj(raw)   # [B, out_dim]
        return raw, proj
