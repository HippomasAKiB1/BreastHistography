class TransformerMagFusion(nn.Module):
    """
    Phase 2.2: Full Transformer encoder over 4 magnification tokens.
    Adds a learnable [CLS] token whose output is the fused representation.
    Uses 4 encoder layers with 8 heads — deeper than original single-attn.
    """
    def __init__(self, feat_dim=512, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, 5, feat_dim))  # 4 mags + CLS
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=num_heads,
            dim_feedforward=feat_dim*4, dropout=dropout,
            batch_first=True, norm_first=True   # pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(feat_dim)

    def _last_layer_with_attn(self, layer, x):
        """
        Re-implements nn.TransformerEncoderLayer's pre-norm (norm_first=True)
        forward pass for a single layer, but requests real attention weights
        from the internal MultiheadAttention call (need_weights=True).
        nn.TransformerEncoder normally hides these because it uses a fast
        path with need_weights=False for speed, which is why the previous
        implementation could only return a hardcoded zero placeholder here.
        """
        x_norm = layer.norm1(x)
        attn_out, attn_w = layer.self_attn(
            x_norm, x_norm, x_norm,
            need_weights=True, average_attn_weights=True)   # [B, tgt, src]
        x = x + layer.dropout1(attn_out)
        x = x + layer._ff_block(layer.norm2(x))
        return x, attn_w

    def forward(self, feats):
        """
        feats: list of 4 tensors [B, feat_dim]
        returns: cls_out [B, feat_dim], attn_weights [B, 4, 4] (real weights,
                 mag-token-to-mag-token self-attention from the last layer)
        """
        B = feats[0].size(0)
        x = torch.stack(feats, dim=1)                        # [B, 4, D]
        cls = self.cls_token.expand(B, -1, -1)              # [B, 1, D]
        x   = torch.cat([cls, x], dim=1)                    # [B, 5, D]
        x   = (x + self.pos_embed).float()                   # fp32 for stability
        # Run all-but-last layers normally, then re-run the last layer while
        # capturing its attention weights.
        for layer in self.encoder.layers[:-1]:
            x = layer(x)
        x, attn_full = self._last_layer_with_attn(self.encoder.layers[-1], x)
        x = x.to(dtype=feats[0].dtype)
        cls_out = self.norm(x[:, 0])                        # [B, D] — CLS token
        # attn_full is [B, 5, 5] (CLS + 4 mag tokens); keep mag-to-mag block
        attn_w = attn_full[:, 1:, 1:].detach()               # [B, 4, 4]
        return cls_out, attn_w


class MAGNET(nn.Module):
    """
    MAGNET v2 — upgraded for performance:
    - Stronger backbones (EfficientNet-B2/B1, ResNet50, EfficientNet-B0)
    - Transformer encoder fusion with CLS token (4 layers, 8 heads)
    - Learnable attention GNN
    - 3-head ensemble binary classifier (averaged at inference)
    - Focal Loss compatible outputs
    """
    def __init__(self, feat_dim=FEAT_DIM, num_binary=2, num_subtype=8,
                 pretrained=True, use_gnn=True, use_aux=True):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_aux = use_aux

        # ── Backbone branches ───────────────────────────────────────────────
        self.branches = nn.ModuleDict({
            mag: BranchExtractor(BACKBONES[mag], BACKBONE_OUT_DIM[mag],
                                 feat_dim, pretrained)
            for mag in MAGNIFICATIONS
        })

        # ── Transformer fusion ──────────────────────────────────────────────
        self.fusion = TransformerMagFusion(feat_dim, num_heads=8, num_layers=4)

        # ── GNN ────────────────────────────────────────────────────────────
        if use_gnn:
            self.gnn = BatchGraphNet(feat_dim, hidden_dim=256)

        # ── Multi-head ensemble binary classifier (Phase 2.4) ───────────────
        def _bin_head():
            # FIX: BatchNorm1d with BATCH_SIZE=4 gives a very noisy running
            # variance estimate early in training; dividing by a
            # near-zero variance blows activations up, which overflows to
            # inf/nan under fp16 autocast within the first few steps. This
            # is almost certainly the source of the "NaN/Inf grad norm"
            # errors. LayerNorm normalises per-sample instead of across the
            # batch, so it's stable regardless of batch size.
            return nn.Sequential(
                nn.Linear(feat_dim, 256), nn.LayerNorm(256), nn.GELU(),
                nn.Dropout(0.4), nn.Linear(256, num_binary))

        self.binary_heads = nn.ModuleList([_bin_head() for _ in range(3)])

        # ── Subtype head (auxiliary regulariser) ────────────────────────────
        self.subtype_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.LayerNorm(256), nn.GELU(),   # FIX: see _bin_head note
            nn.Dropout(0.4), nn.Linear(256, num_subtype))

        # ── Uncertainty head ────────────────────────────────────────────────
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid())

        # ── Auxiliary per-branch heads ──────────────────────────────────────
        if use_aux:
            self.aux_heads = nn.ModuleDict({
                mag: nn.Linear(BACKBONE_OUT_DIM[mag], num_binary)
                for mag in MAGNIFICATIONS
            })

    @property
    def binary_head(self):
        """Compatibility shim: returns the first head (used by GradCAM etc.)"""
        return self.binary_heads[0]

    def _ensemble_binary(self, fused):
        """Average logits across the 3 binary heads."""
        return torch.stack([h(fused) for h in self.binary_heads], dim=0).mean(0)

    def forward(self, x40, x100, x200, x400, return_attn=False):
        inputs = {'40X':x40,'100X':x100,'200X':x200,'400X':x400}
        raws, projs, aux_logits = {}, {}, {}
        for mag in MAGNIFICATIONS:
            raw, proj = self.branches[mag](inputs[mag])
            raws[mag]=raw; projs[mag]=proj
            if self.use_aux:
                aux_logits[mag] = self.aux_heads[mag](raw)
        fused, attn_w = self.fusion([projs[m] for m in MAGNIFICATIONS])
        if self.use_gnn:
            fused = self.gnn(fused)
        bl  = self._ensemble_binary(fused)
        sl  = self.subtype_head(fused)
        unc = self.uncertainty_head(fused)
        out = (bl, sl, unc, list(aux_logits.values()) if self.use_aux else [])
        return (*out, attn_w) if return_attn else out

    @torch.no_grad()
    def mc_uncertainty(self, x40, x100, x200, x400, n_passes=20):
        self.train()
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
        probs = []
        for _ in range(n_passes):
            logits, *_ = self(x40, x100, x200, x400)
            probs.append(F.softmax(logits, dim=1).unsqueeze(0))
        self.eval()
        probs = torch.cat(probs, dim=0)
        return probs.mean(0), probs.var(0)

print('MAGNET v2 defined (TransformerFusion + 3-head ensemble).')
