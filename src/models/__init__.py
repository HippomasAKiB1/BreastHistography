"""
Neural network architectures and model components for BreastHistography (MAGNET).
"""

from .backbones import BACKBONES, BACKBONE_OUT_DIM, MAGNIFICATIONS, BranchExtractor
from .fusion import CrossMagFusion
from .graph_net import BatchGraphNet, SparseGraphConv
from .magnet import MAGNET
from .baselines import ConcatFusionBaseline, SingleMagBaseline

__all__ = [
    "BACKBONES",
    "BACKBONE_OUT_DIM",
    "MAGNIFICATIONS",
    "BranchExtractor",
    "CrossMagFusion",
    "BatchGraphNet",
    "SparseGraphConv",
    "MAGNET",
    "ConcatFusionBaseline",
    "SingleMagBaseline",
]
