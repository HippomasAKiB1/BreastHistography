"""
Unit tests for neural network architectures in src.models.
"""

import pytest
import torch

from src.models import (
    BranchExtractor,
    CrossMagFusion,
    BatchGraphNet,
    MAGNET,
    ConcatFusionBaseline,
    SingleMagBaseline,
)


def test_branch_extractor():
    branch = BranchExtractor(backbone_name="resnet34", in_dim=512, out_dim=256, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    raw, proj = branch(x)
    assert raw.shape == (2, 512), f"Expected raw shape (2, 512), got {raw.shape}"
    assert proj.shape == (2, 256), f"Expected proj shape (2, 256), got {proj.shape}"


def test_cross_mag_fusion():
    fusion = CrossMagFusion(feat_dim=128, num_heads=4)
    feats = [torch.randn(2, 128) for _ in range(4)]
    fused, attn_w = fusion(feats)
    assert fused.shape == (2, 128), f"Expected fused shape (2, 128), got {fused.shape}"
    assert attn_w.shape == (2, 4, 4), f"Expected attention shape (2, 4, 4), got {attn_w.shape}"


def test_batch_graph_net():
    gnn = BatchGraphNet(feat_dim=128, hidden_dim=64, num_layers=2)
    h = torch.randn(4, 128)
    out = gnn(h)
    assert out.shape == (4, 128), f"Expected GNN output (4, 128), got {out.shape}"


def test_magnet_forward():
    model = MAGNET(feat_dim=128, num_binary=2, num_subtype=8, pretrained=False, use_gnn=True, use_aux=True)
    model.eval()
    x40 = torch.randn(2, 3, 224, 224)
    x100 = torch.randn(2, 3, 224, 224)
    x200 = torch.randn(2, 3, 224, 224)
    x400 = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        bin_logits, sub_logits, uncert, aux_logits = model(x40, x100, x200, x400)

    assert bin_logits.shape == (2, 2), f"Expected bin_logits shape (2, 2), got {bin_logits.shape}"
    assert sub_logits.shape == (2, 8), f"Expected sub_logits shape (2, 8), got {sub_logits.shape}"
    assert uncert.shape == (2, 1), f"Expected uncertainty shape (2, 1), got {uncert.shape}"
    assert len(aux_logits) == 4, f"Expected 4 auxiliary logits, got {len(aux_logits)}"


def test_concat_fusion_baseline():
    model = ConcatFusionBaseline(proj_dim=64, num_binary=2, pretrained=False)
    model.eval()
    x40 = torch.randn(2, 3, 224, 224)
    x100 = torch.randn(2, 3, 224, 224)
    x200 = torch.randn(2, 3, 224, 224)
    x400 = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        logits = model(x40, x100, x200, x400)
    assert logits.shape == (2, 2), f"Expected logits shape (2, 2), got {logits.shape}"


def test_single_mag_baseline():
    model = SingleMagBaseline(mag="200X", num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 2), f"Expected logits shape (2, 2), got {logits.shape}"
