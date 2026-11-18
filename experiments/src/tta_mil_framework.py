"""
tta_mil_framework.py: High-performance, mathematically rigorous implementation
of Test-Time Adaptation for Multi-Instance Learning.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class GatedAttentionPooling(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=256):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        # h: [B, N, D]
        V = self.attention_V(h)
        U = self.attention_U(h)
        weights = self.attention_w(V * U)  # [B, N, 1]
        attn = F.softmax(weights, dim=1)
        z = torch.sum(attn * h, dim=1)     # [B, D]
        return z, attn

class ABMILClassifier(nn.Module):
    def __init__(self, in_dim=1024, num_classes=2, hidden_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.pooling = GatedAttentionPooling(in_dim, hidden_dim)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        # x: [B, N, D]
        x_norm = self.norm(x)
        z, attn = self.pooling(x_norm)
        logits = self.classifier(z)
        return logits, z, attn

class FastTTAMILAdapter:
    def __init__(self, model, lr=1e-3, source_cov=None, source_prototypes=None):
        self.model = model
        self.lr = lr
        self.source_cov = source_cov
        self.source_prototypes = source_prototypes
        # Cache initial state for microsecond restoration
        self.init_state = {k: v.clone() for k, v in model.state_dict().items()}

    def reset(self):
        self.model.load_state_dict(self.init_state)

    def adapt_bag(self, feat, method="no_tta", steps=3):
        """
        Adapts on a single bag [1, N, D], resets weights after prediction.
        """
        if method == "no_tta":
            with torch.no_grad():
                logits, _, _ = self.model(feat)
            return logits

        self.reset()
        self.model.train()

        if method in ["tent_1", "tent_10"]:
            max_steps = 1 if method == "tent_1" else steps
            params = [p for n, p in self.model.named_parameters() if 'norm' in n]
            for p in self.model.parameters(): p.requires_grad = False
            for p in params: p.requires_grad = True
            optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9)

            for _ in range(max_steps):
                optimizer.zero_grad()
                logits, _, _ = self.model(feat)
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                entropy.backward()
                optimizer.step()

        elif method == "coral_5":
            params = [p for n, p in self.model.named_parameters() if 'norm' in n]
            for p in self.model.parameters(): p.requires_grad = False
            for p in params: p.requires_grad = True
            optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9)

            for _ in range(steps):
                optimizer.zero_grad()
                logits, _, _ = self.model(feat)
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()

                patches = feat.squeeze(0)
                N, D = patches.shape
                patches_norm = self.model.norm(patches)
                mean_t = patches_norm.mean(dim=0, keepdim=True)
                cov_t = (patches_norm - mean_t).T @ (patches_norm - mean_t) / (N - 1 + 1e-8)

                coral_loss = torch.norm(self.source_cov.to(cov_t.device) - cov_t, p='fro') / (4.0 * D * D)
                loss = entropy + 0.5 * coral_loss
                loss.backward()
                optimizer.step()

        elif method == "hat_mil":
            # Head-Aware Adaptation: updates LayerNorm AND classifier
            params = [p for n, p in self.model.named_parameters() if 'norm' in n or 'classifier' in n]
            for p in self.model.parameters(): p.requires_grad = False
            for p in params: p.requires_grad = True
            optimizer = torch.optim.SGD(params, lr=self.lr * 0.5, momentum=0.9)

            for _ in range(steps):
                optimizer.zero_grad()
                logits, z, _ = self.model(feat)
                probs = F.softmax(logits, dim=-1)
                max_prob, pseudo_label = torch.max(probs, dim=-1)

                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                selected_proto = self.source_prototypes[pseudo_label].to(z.device)
                proto_loss = F.mse_loss(z, selected_proto)

                loss = entropy + 0.4 * proto_loss
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            self.model.eval()
            logits, _, _ = self.model(feat)
        self.reset()
        return logits
