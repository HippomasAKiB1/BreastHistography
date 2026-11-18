class BranchExtractor(nn.Module):
    """Single pretrained backbone + linear projection."""

    def __init__(self, backbone_name: str, in_dim: int, out_dim: int, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          num_classes=0, global_pool='avg')
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        # Cache raw features for auxiliary head
        self._raw_dim = in_dim

    def forward(self, x):
        raw = self.backbone(x)          # [B, in_dim]
        proj = self.proj(raw)           # [B, out_dim]
        return raw, proj

print('BranchExtractor defined.')