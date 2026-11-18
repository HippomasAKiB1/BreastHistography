import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseGraphConv(nn.Module):
    """
    Pure PyTorch graph convolution layer: h' = ReLU(D^{-0.5} A D^{-0.5} H W).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        return F.relu(adj @ self.W(h))

class BatchGraphNet(nn.Module):
    """
    Treats each batch of patient embeddings as nodes in a graph.
    Builds on-the-fly cosine similarity adjacency with self-loops,
    applies 2 GCN layers, and adds a residual connection back to input.
    """
    def __init__(self, feat_dim: int = 512, hidden_dim: int = 256, num_layers: int = 2,
                 sim_threshold: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.threshold = sim_threshold
        layers = []
        in_d = feat_dim
        for _ in range(num_layers):
            layers.append(SparseGraphConv(in_d, hidden_dim))
            in_d = hidden_dim
        self.gcn_layers = nn.ModuleList(layers)
        self.drop       = nn.Dropout(dropout)
        self.readout    = nn.Linear(hidden_dim, feat_dim)
        self.norm       = nn.LayerNorm(feat_dim)

    def _build_adj(self, h: torch.Tensor) -> torch.Tensor:
        norm = F.normalize(h.detach(), p=2, dim=1)
        sim  = torch.mm(norm, norm.t())
        mask = (sim > self.threshold).float()
        mask.fill_diagonal_(1.0)
        deg  = mask.sum(dim=1, keepdim=True).clamp(min=1)
        d_inv_sqrt = deg.pow(-0.5)
        adj  = mask * d_inv_sqrt * d_inv_sqrt.t()
        return adj

    def forward(self, h: torch.Tensor):
        adj = self._build_adj(h)
        out = h
        for layer in self.gcn_layers:
            out = self.drop(layer(out, adj))
        return self.norm(h + self.readout(out))
