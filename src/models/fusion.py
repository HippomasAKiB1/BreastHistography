import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossMagFusion(nn.Module):
    """
    Multi-head self-attention across 4 magnification tokens.
    Learns dynamic per-magnification scaling weights and aggregates
    representations with a residual feed-forward network.
    """
    def __init__(self, feat_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert feat_dim % num_heads == 0, "feat_dim must be divisible by num_heads"
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.mag_scale = nn.Parameter(torch.ones(4))
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim),
        )
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

    def forward(self, feats: list):
        """
        feats: list of 4 tensors, each [B, feat_dim]
        returns: fused [B, feat_dim], attn_weights [B, 4, 4]
        """
        x = torch.stack(feats, dim=1)                      # [B, 4, feat_dim]
        attn_out, attn_w = self.attn(x, x, x)              # [B, 4, feat_dim], [B, 4, 4]
        x = self.norm1(x + attn_out)                       # residual
        w = F.softmax(self.mag_scale, dim=0)               # [4]
        fused = (x * w.unsqueeze(0).unsqueeze(-1)).sum(1)  # [B, feat_dim]
        fused = self.norm2(fused + self.ffn(fused))        # FFN + residual
        return fused, attn_w
