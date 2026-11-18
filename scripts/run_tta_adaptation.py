#!/usr/bin/env python3
"""
scripts/run_tta_adaptation.py: Benchmark Test-Time Adaptation (TTA) algorithms
on patient feature representations.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tta import FastTTAMILAdapter
from src.eval import compute_classification_metrics
from src.utils import set_seed, save_json


class GatedAttentionPooling(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=256):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        v = self.attention_V(h)
        u = self.attention_U(h)
        weights = self.attention_w(v * u)
        attn = F.softmax(weights, dim=1)
        z = torch.sum(attn * h, dim=1)
        return z, attn


class ABMILClassifier(nn.Module):
    def __init__(self, in_dim=1024, num_classes=2, hidden_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.pooling = GatedAttentionPooling(in_dim, hidden_dim)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x_norm = self.norm(x)
        z, attn = self.pooling(x_norm)
        logits = self.classifier(z)
        return logits, z, attn


def run_tta_benchmark(output_json: str = None, seed: int = 42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running TTA Benchmark on: {device}")

    # Generate or load bag representations
    # Simulating standard target domain test bags (e.g. 50 target bags with covariate shift)
    in_dim = 1024
    num_bags = 50
    model = ABMILClassifier(in_dim=in_dim, num_classes=2).to(device)

    # Synthetic source prototypes
    source_prototypes = torch.randn(2, in_dim).to(device)
    source_cov = torch.eye(in_dim).to(device)

    adapter = FastTTAMILAdapter(
        model=model,
        lr=1e-3,
        source_cov=source_cov,
        source_prototypes=source_prototypes,
    )

    methods = ["no_tta", "tent_1", "tent_10", "coral_5", "hat_mil"]
    results = {}

    print("\nExecuting Test-Time Adaptation across strategies:")
    for method in methods:
        preds = []
        for b in range(num_bags):
            # Target bag under stain/domain shift
            bag_len = np.random.randint(15, 30)
            feat = torch.randn(1, bag_len, in_dim).to(device) + 0.35
            logits = adapter.adapt_bag(feat, method=method, steps=3)
            prob = F.softmax(logits, dim=-1)[:, 1].item()
            preds.append(prob)

        pred_labels = (np.array(preds) >= 0.5).astype(int)
        collapse_rate = float(np.mean(pred_labels == 1))
        results[method] = {
            "mean_prob": float(np.mean(preds)),
            "std_prob": float(np.std(preds)),
            "positive_fraction": collapse_rate,
        }
        print(f"  • {method:10s} -> Positive Fraction: {collapse_rate:.3f}, Mean Conf: {np.mean(preds):.3f}")

    if output_json:
        save_json(results, output_json)
        print(f"\nSaved TTA benchmark to: {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TTA Adaptation benchmark")
    parser.add_argument("--output", type=str, default="results/tta_benchmark_run.json")
    args = parser.parse_args()
    run_tta_benchmark(args.output)
