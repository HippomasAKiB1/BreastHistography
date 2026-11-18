class MAGNET(nn.Module):
    """
    Multi-scale Attention-based Graph Network with Ensemble Transfer Learning.

    Architecture:
      • 4 lightweight pretrained backbones (one per magnification)
      • Feature projection to shared feat_dim
      • Cross-magnification multi-head attention fusion
      • Batch graph network for inter-sample relationship modelling
      • Dual heads: binary (B/M) + subtype (8-class)
      • Auxiliary per-branch binary heads (deep supervision)
      • Uncertainty head (MC-Dropout compatible)
    """

    def __init__(self, feat_dim=FEAT_DIM, num_binary=2, num_subtype=8,
                 pretrained=True, use_gnn=True, use_aux=True):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_aux = use_aux

        # ── Backbone branches ──────────────────────────────────────────────
        self.branches = nn.ModuleDict()
        for mag in MAGNIFICATIONS:
            self.branches[mag] = BranchExtractor(
                BACKBONES[mag], BACKBONE_OUT_DIM[mag], feat_dim, pretrained
            )

        # ── Cross-magnification fusion ─────────────────────────────────────
        self.fusion = CrossMagFusion(feat_dim, num_heads=8)

        # ── Graph network ──────────────────────────────────────────────────
        if use_gnn:
            self.gnn = BatchGraphNet(feat_dim, hidden_dim=256)

        # ── Main classification heads ──────────────────────────────────────
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

        # ── Uncertainty head ───────────────────────────────────────────────
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # ── Auxiliary per-branch heads (deep supervision) ──────────────────
        if use_aux:
            self.aux_heads = nn.ModuleDict({
                mag: nn.Linear(BACKBONE_OUT_DIM[mag], num_binary)
                for mag in MAGNIFICATIONS
            })

    def forward(self, x40, x100, x200, x400, return_attn=False):
        inputs = {'40X': x40, '100X': x100, '200X': x200, '400X': x400}
        raws, projs, aux_logits = {}, {}, {}

        for mag in MAGNIFICATIONS:
            raw, proj = self.branches[mag](inputs[mag])
            raws[mag]  = raw
            projs[mag] = proj
            if self.use_aux:
                aux_logits[mag] = self.aux_heads[mag](raw)

        # Fusion
        feat_list = [projs[m] for m in MAGNIFICATIONS]
        fused, attn_w = self.fusion(feat_list)            # [B, feat_dim], [B, 4, 4]

        # GNN
        if self.use_gnn:
            fused = self.gnn(fused)

        binary_logits  = self.binary_head(fused)
        subtype_logits = self.subtype_head(fused)
        uncertainty    = self.uncertainty_head(fused)     # [B, 1]

        out = (binary_logits, subtype_logits, uncertainty,
               list(aux_logits.values()) if self.use_aux else [])
        if return_attn:
            return (*out, attn_w)
        return out

    @torch.no_grad()
    def mc_uncertainty(self, x40, x100, x200, x400, n_passes=20):
        """Monte-Carlo Dropout uncertainty estimate (enable dropout at inference)."""
        self.train()       # keeps dropout active
        probs = []
        for _ in range(n_passes):
            logits, *_ = self(x40, x100, x200, x400)
            probs.append(F.softmax(logits, dim=1).unsqueeze(0))
        self.eval()
        probs = torch.cat(probs, dim=0)    # [n_passes, B, 2]
        mean  = probs.mean(0)              # [B, 2]
        var   = probs.var(0)               # [B, 2]  — epistemic uncertainty
        return mean, var

print('MAGNET defined.')