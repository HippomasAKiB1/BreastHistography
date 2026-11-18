#!/usr/bin/env python3
"""
scripts/evaluate_checkpoints.py: Reproducible evaluation runner for MAGNET checkpoints
and baseline models on the BreaKHis cohort.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import MAGNET, ConcatFusionBaseline, SingleMagBaseline
from src.data import (
    collect_patient_info,
    get_fixed_train_val_test_split,
    MultiMagDataset,
    SingleMagDataset,
    get_eval_transforms,
)
from src.eval import compute_classification_metrics
from src.utils import set_seed, load_yaml, save_json, load_checkpoint


def evaluate(
    model_type: str = "magnet",
    checkpoint_path: str = None,
    data_dir: str = "data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast",
    batch_size: int = 4,
    num_repeats: int = 5,
    seed: int = 42,
    output_json: str = None,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    # Index dataset
    patient_info = collect_patient_info(data_dir)
    _, _, test_pids = get_fixed_train_val_test_split(patient_info, seed=seed)
    print(f"Test cohort: {len(test_pids)} patients")

    # Instantiate model
    if model_type == "magnet":
        model = MAGNET(pretrained=False, use_gnn=True, use_aux=True)
    elif model_type == "concat":
        model = ConcatFusionBaseline(pretrained=False)
    elif model_type.startswith("single_"):
        mag = model_type.split("_")[1].upper()
        model = SingleMagBaseline(mag=mag, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        state = load_checkpoint(checkpoint_path, device=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
    else:
        print("Note: Running with initialized model weights (zero-shot or unweighted mode).")

    model = model.to(device)
    model.eval()

    all_pass_probs = []
    ground_truth = [patient_info[p]["binary"] for p in test_pids]

    for pass_idx in range(num_repeats):
        ds = MultiMagDataset(
            patient_ids=test_pids,
            patient_info=patient_info,
            stain_norm=True,
            seed=1000 + pass_idx * 17,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        pass_probs = []
        with torch.no_grad():
            for batch in loader:
                x40, x100, x200, x400, _, _, _ = batch
                x40, x100, x200, x400 = x40.to(device), x100.to(device), x200.to(device), x400.to(device)
                out = model(x40, x100, x200, x400)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                pass_probs.extend(probs)
        all_pass_probs.append(pass_probs)

    avg_probs = np.mean(all_pass_probs, axis=0)
    metrics = compute_classification_metrics(ground_truth, avg_probs)

    print("\n" + "=" * 50)
    print(f"Evaluation Results [{model_type.upper()}]:")
    print("=" * 50)
    print(f"  • Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  • Precision:       {metrics['precision']:.4f}")
    print(f"  • Recall:          {metrics['recall']:.4f}")
    print(f"  • Macro-F1:        {metrics['f1_macro']:.4f}")
    print(f"  • AUROC:           {metrics['auroc']:.4f}" if metrics['auroc'] is not None else "  • AUROC:           N/A")
    print(f"  • ECE (Calibration): {metrics['ece']:.4f}")
    print(f"  • Brier Score:     {metrics['brier_score']:.4f}")
    print("=" * 50)

    if output_json:
        save_json(metrics, output_json)
        print(f"Saved evaluation metrics to: {output_json}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MAGNET & Baselines")
    parser.add_argument("--model", type=str, default="magnet", choices=["magnet", "concat", "single_40x", "single_100x", "single_200x", "single_400x"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=str, default=None, help="Path to output json file")
    args = parser.parse_args()

    evaluate(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_repeats=args.repeats,
        output_json=args.output,
    )
