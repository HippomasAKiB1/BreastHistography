import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import BranchExtractor
from src.models.fusion import CrossMagFusion
from src.models.graph_net import BatchGraphNet

MAGNIFICATIONS = ['40X', '100X', '200X', '400X']
BACKBONES = {
    '40X' : 'mobilenetv3_large_100',
    '100X': 'efficientnet_b0',
    '200X': 'resnet34',
    '400X': 'mobilenetv3_small_100',
}
BACKBONE_OUT_DIM = {
    '40X' : 1280,
    '100X': 1280,
    '200X': 512,
    '400X': 1024,
}

class MAGNET(nn.Module):
    """
    Multi-scale Attention-based Graph Network with Ensemble Transfer Learning.
    
    Processes 4 magnification tokens (40X, 100X, 200X, 400X), applies cross-magnification
    multi-head attention fusion, inter-sample batch graph convolution, and dual classification
    heads for binary diagnostic triage and histological subtyping.
    """
    def __init__(self, feat_dim: int = 512, num_binary: int = 2, num_subtype: int = 8,
                 pretrained: bool = False, use_gnn: bool = True, use_aux: bool = True):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_aux = use_aux

        self.branches = nn.ModuleDict()
        for mag in MAGNIFICATIONS:
            self.branches[mag] = BranchExtractor(
                BACKBONES[mag], BACKBONE_OUT_DIM[mag], feat_dim, pretrained
            )

        self.fusion = CrossMagFusion(feat_dim, num_heads=8)

        if use_gnn:
            self.gnn = BatchGraphNet(feat_dim, hidden_dim=256)

        def _head(in_d, out_d):
            return nn.Sequential(
                nn.Linear(in_d, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(256, out_d),
            )

        self.binary_head  = _head(feat_dim, num_binary)
        self.subtype_head = _head(feat_dim, num_subtype)

        self.uncertainty_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        if use_aux:
            self.aux_heads = nn.ModuleDict({
                mag: nn.Linear(BACKBONE_OUT_DIM[mag], num_binary)
                for mag in MAGNIFICATIONS
            })

    def forward(self, x40: torch.Tensor, x100: torch.Tensor, x200: torch.Tensor, x400: torch.Tensor, return_attn: bool = False):
        inputs = {'40X': x40, '100X': x100, '200X': x200, '400X': x400}
        raws, projs, aux_logits = {}, {}, {}

        for mag in MAGNIFICATIONS:
            raw, proj = self.branches[mag](inputs[mag])
            raws[mag]  = raw
            projs[mag] = proj
            if self.use_aux:
                aux_logits[mag] = self.aux_heads[mag](raw)

        feat_list = [projs[m] for m in MAGNIFICATIONS]
        fused, attn_w = self.fusion(feat_list)

        if self.use_gnn:
            fused = self.gnn(fused)

        binary_logits  = self.binary_head(fused)
        subtype_logits = self.subtype_head(fused)
        uncertainty    = self.uncertainty_head(fused)

        out = (binary_logits, subtype_logits, uncertainty,
               list(aux_logits.values()) if self.use_aux else [])
        if return_attn:
            return (*out, attn_w)
        return out

class MAGNETLoss(nn.Module):
    """
    Weighted multi-task loss: L = alpha*L_bin + beta*L_sub + gamma*L_aux + delta*L_unc
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.3, delta: float = 0.05,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        w = class_weights if class_weights is not None else torch.tensor([0.69, 0.31])
        self.ce_binary  = nn.CrossEntropyLoss(weight=w)
        self.ce_subtype = nn.CrossEntropyLoss()
        self.ce_aux     = nn.CrossEntropyLoss()
        self.mse        = nn.MSELoss()

    def forward(self, binary_logits, subtype_logits, uncertainty, aux_logits, y_bin, y_sub):
        self.ce_binary.weight = self.ce_binary.weight.to(binary_logits.device)
        L_bin = self.ce_binary(binary_logits, y_bin)
        L_sub = self.ce_subtype(subtype_logits, y_sub)

        L_aux = torch.tensor(0.0, device=binary_logits.device)
        if aux_logits:
            for aux in aux_logits:
                L_aux = L_aux + self.ce_aux(aux, y_bin)
            L_aux = L_aux / len(aux_logits)

        with torch.no_grad():
            conf = F.softmax(binary_logits, dim=1).max(dim=1).values
            target_unc = (1.0 - conf).unsqueeze(1)
        L_unc = self.mse(uncertainty, target_unc)

        total = (self.alpha * L_bin + self.beta * L_sub +
                 self.gamma * L_aux + self.delta * L_unc)
        return total, {'binary': L_bin.item(), 'subtype': L_sub.item(), 'aux': L_aux.item(), 'unc': L_unc.item()}
