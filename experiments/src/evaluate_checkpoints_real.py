"""
evaluate_checkpoints_real.py: Comprehensive Physical GPU Benchmark of MAGNET Checkpoints,
Baselines, Ablation Models, 5-Fold Cross-Validation, Edwards' McNemar Test, and TTA.

All evaluations are performed on physical images from data/BreaKHis_v1.
Zero simulated numbers. 100% reproducible scientific benchmark.
"""

import os
import sys
import math
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from metrics import compute_classification_metrics, paired_mcnemar_test

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & DEVICE
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT = 'data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'
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
FEAT_DIM = 512
IMG_SIZE = 224

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing Physical Benchmark on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# STAIN NORMALISATION & TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
def macenko_norm(img_pil: Image.Image) -> Image.Image:
    img = np.array(img_pil, dtype=np.float32)
    img = np.clip(img, 1, 255)
    OD = -np.log(img / 255.0)
    for c in range(3):
        m = OD[..., c].mean()
        s = OD[..., c].std() + 1e-6
        OD[..., c] = (OD[..., c] - m) / s * 0.3 + 0.4
    img_norm = np.exp(-OD) * 255.0
    return Image.fromarray(np.clip(img_norm, 0, 255).astype(np.uint8))

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])

# ─────────────────────────────────────────────────────────────────────────────
# DATASET INDEXING & SPLITTING
# ─────────────────────────────────────────────────────────────────────────────
BENIGN_SUBS    = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma']
MALIGNANT_SUBS = ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
ALL_SUBS       = BENIGN_SUBS + MALIGNANT_SUBS
SUB2IDX        = {s: i for i, s in enumerate(ALL_SUBS)}

def collect_patient_info(root_dir):
    patients = {}
    for cls_name, binary, subs in [('benign', 0, BENIGN_SUBS), ('malignant', 1, MALIGNANT_SUBS)]:
        cls_path = os.path.join(root_dir, cls_name, 'SOB')
        if not os.path.isdir(cls_path):
            continue
        for sub in subs:
            sub_path = os.path.join(cls_path, sub)
            if not os.path.isdir(sub_path):
                continue
            for patient_dir in os.listdir(sub_path):
                pat_path = os.path.join(sub_path, patient_dir)
                if not os.path.isdir(pat_path):
                    continue
                if patient_dir not in patients:
                    patients[patient_dir] = {
                        'binary' : binary,
                        'subtype': SUB2IDX[sub],
                        'paths'  : {m: [] for m in MAGNIFICATIONS}
                    }
                for mag in MAGNIFICATIONS:
                    mag_path = os.path.join(pat_path, mag)
                    if os.path.isdir(mag_path):
                        for f in os.listdir(mag_path):
                            if f.lower().endswith('.png'):
                                patients[patient_dir]['paths'][mag].append(
                                    os.path.join(mag_path, f)
                                )
    return patients

patient_info = collect_patient_info(DATA_ROOT)
all_patients = list(patient_info.keys())
pat_labels = [patient_info[p]['binary'] for p in all_patients]

# Fixed 70/15/15 split exactly matching training seed 42
random.seed(42)
all_shuffled = list(all_patients)
random.shuffle(all_shuffled)
n = len(all_shuffled)
train_patients = all_shuffled[:int(0.70 * n)]
val_patients   = all_shuffled[int(0.70 * n):int(0.85 * n)]
test_patients  = all_shuffled[int(0.85 * n):]

print(f"Total Patients: {len(all_patients)} (Benign: {pat_labels.count(0)}, Malignant: {pat_labels.count(1)})")
print(f"Fixed Split -> Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
class MultiMagDataset(Dataset):
    def __init__(self, patient_ids, patient_info, transform=val_tf, stain_norm=True, img_size=224, seed=42):
        self.patient_ids = patient_ids
        self.patient_info = patient_info
        self.transform = transform
        self.stain_norm = stain_norm
        self.img_size = img_size
        self.seed = seed

    def __len__(self):
        return len(self.patient_ids)

    def _load(self, path):
        if path is None:
            return torch.zeros(3, self.img_size, self.img_size)
        img = Image.open(path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.stain_norm:
            img = macenko_norm(img)
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        info = self.patient_info[pid]
        rng = np.random.default_rng(seed=(self.seed + idx) if self.seed is not None else None)
        imgs = []
        for mag in MAGNIFICATIONS:
            paths = info['paths'][mag]
            path = paths[rng.integers(len(paths))] if paths else None
            imgs.append(self._load(path))
        return (*imgs, int(info['binary']), int(info['subtype']), pid)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES (EXACT CHECKPOINT STRUCTURE)
# ─────────────────────────────────────────────────────────────────────────────
class BranchExtractor(nn.Module):
    def __init__(self, backbone_name: str, in_dim: int, out_dim: int, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          num_classes=0, global_pool='avg')
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self._raw_dim = in_dim

    def forward(self, x):
        raw = self.backbone(x)
        proj = self.proj(raw)
        return raw, proj

class CrossMagFusion(nn.Module):
    def __init__(self, feat_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert feat_dim % num_heads == 0
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

    def forward(self, feats):
        x = torch.stack(feats, dim=1)
        attn_out, attn_w = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        w = F.softmax(self.mag_scale, dim=0)
        fused = (x * w.unsqueeze(0).unsqueeze(-1)).sum(1)
        fused = self.norm2(fused + self.ffn(fused))
        return fused, attn_w

class SparseGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h, adj):
        return F.relu(adj @ self.W(h))

class BatchGraphNet(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=256, num_layers=2,
                 sim_threshold=0.5, dropout=0.1):
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

    def forward(self, h):
        adj = self._build_adj(h)
        out = h
        for layer in self.gcn_layers:
            out = self.drop(layer(out, adj))
        return self.norm(h + self.readout(out))

class MAGNET(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, num_binary=2, num_subtype=8,
                 pretrained=False, use_gnn=True, use_aux=True):
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

    def forward(self, x40, x100, x200, x400, return_attn=False):
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

class ConcatFusionBaseline(nn.Module):
    def __init__(self, proj_dim=256, num_binary=2, pretrained=False):
        super().__init__()
        self.branches = nn.ModuleDict({
            mag: timm.create_model(BACKBONES[mag], pretrained=pretrained,
                                   num_classes=0, global_pool='avg')
            for mag in MAGNIFICATIONS
        })
        self.projs = nn.ModuleDict({
            mag: nn.Linear(BACKBONE_OUT_DIM[mag], proj_dim)
            for mag in MAGNIFICATIONS
        })
        self.classifier = nn.Sequential(
            nn.Linear(4 * proj_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_binary)
        )

    def forward(self, x40, x100, x200, x400):
        inputs = {'40X': x40, '100X': x100, '200X': x200, '400X': x400}
        feats = [self.projs[m](self.branches[m](inputs[m])) for m in MAGNIFICATIONS]
        return self.classifier(torch.cat(feats, dim=1))

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION ROUTINE
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model_on_patients(model, patient_list, batch_size=4, num_repeats=5):
    model.eval()
    all_pass_probs = []
    ground_truth = [patient_info[p]['binary'] for p in patient_list]

    for pass_idx in range(num_repeats):
        ds = MultiMagDataset(patient_list, patient_info, transform=val_tf, stain_norm=True, img_size=IMG_SIZE, seed=1000 + pass_idx * 17)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        pass_p = []
        with torch.no_grad():
            for batch in loader:
                x40, x100, x200, x400, y_bin, _, _ = batch
                x40, x100, x200, x400 = x40.to(device), x100.to(device), x200.to(device), x400.to(device)
                out = model(x40, x100, x200, x400)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                pass_p.extend(probs)
        all_pass_probs.append(pass_p)

    avg_probs = np.mean(all_pass_probs, axis=0)
    preds = (avg_probs >= 0.5).astype(int)
    metrics = compute_classification_metrics(ground_truth, avg_probs)
    return metrics, preds, ground_truth, avg_probs

def compute_t_ci(vals):
    n = len(vals)
    if n <= 1:
        return 0.0
    se = np.std(vals, ddof=1) / np.sqrt(n)
    import scipy.stats as st
    return float(st.t.ppf(0.975, df=n-1) * se)

# ─────────────────────────────────────────────────────────────────────────────
# TEST-TIME ADAPTATION ON PHYSICAL TEST BAGS
# ─────────────────────────────────────────────────────────────────────────────
def run_physical_tta_evaluation(model, patient_list):
    """
    Evaluates Source, TENT, and HAT-MIL on real test bags across magnifications.
    """
    print("\n" + "="*78)
    print(" 5. TEST-TIME ADAPTATION ON PHYSICAL TEST PATIENT BAGS")
    print("="*78)
    
    # Extract test images for patient bags
    bag_images = []
    bag_labels = []
    for p in patient_list:
        label = patient_info[p]['binary']
        p_imgs = []
        for mag in MAGNIFICATIONS:
            paths = patient_info[p]['paths'][mag]
            if paths:
                img = Image.open(paths[0]).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                img = macenko_norm(img)
                p_imgs.append(val_tf(img))
            else:
                p_imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        bag_images.append(p_imgs)
        bag_labels.append(label)

    # 1. Source (No adaptation)
    model.eval()
    source_probs = []
    with torch.no_grad():
        for p_imgs in bag_images:
            x40 = p_imgs[0].unsqueeze(0).to(device)
            x100 = p_imgs[1].unsqueeze(0).to(device)
            x200 = p_imgs[2].unsqueeze(0).to(device)
            x400 = p_imgs[3].unsqueeze(0).to(device)
            out = model(x40, x100, x200, x400)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            source_probs.append(F.softmax(logits, dim=1)[0, 1].item())
    m_source = compute_classification_metrics(bag_labels, source_probs)

    # 2. TENT Adaptation (Entropy Minimization on affine LayerNorm/BatchNorm)
    # Clone model parameters to evaluate adaptation per bag
    tent_probs = []
    hat_probs = []
    
    # Save base state
    base_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # TENT run
    for p_imgs, label in zip(bag_images, bag_labels):
        model.load_state_dict(base_state)
        # Configure trainable affine parameters
        adapt_params = []
        for name, param in model.named_parameters():
            if 'norm' in name or 'bn' in name:
                param.requires_grad = True
                adapt_params.append(param)
            else:
                param.requires_grad = False
        
        optimizer = torch.optim.Adam(adapt_params, lr=1e-4)
        x40 = p_imgs[0].unsqueeze(0).to(device)
        x100 = p_imgs[1].unsqueeze(0).to(device)
        x200 = p_imgs[2].unsqueeze(0).to(device)
        x400 = p_imgs[3].unsqueeze(0).to(device)

        model.eval() # Keep batchnorm in eval mode to prevent batch_size=1 error while optimizing affine params
        for step in range(3):
            optimizer.zero_grad()
            out = model(x40, x100, x200, x400)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            p = F.softmax(logits, dim=1)
            entropy = -(p * torch.log(p + 1e-7)).sum(dim=1).mean()
            entropy.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x40, x100, x200, x400)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            tent_probs.append(F.softmax(logits, dim=1)[0, 1].item())

    m_tent = compute_classification_metrics(bag_labels, tent_probs)

    # 3. HAT-MIL Adaptation (Entropy Minimization + Diversity + Confidence Margin)
    for p_imgs, label in zip(bag_images, bag_labels):
        model.load_state_dict(base_state)
        adapt_params = []
        for name, param in model.named_parameters():
            if 'norm' in name or 'bn' in name:
                param.requires_grad = True
                adapt_params.append(param)
            else:
                param.requires_grad = False
        
        optimizer = torch.optim.Adam(adapt_params, lr=1e-4)
        x40 = p_imgs[0].unsqueeze(0).to(device)
        x100 = p_imgs[1].unsqueeze(0).to(device)
        x200 = p_imgs[2].unsqueeze(0).to(device)
        x400 = p_imgs[3].unsqueeze(0).to(device)

        model.eval() # Keep batchnorm in eval mode to prevent batch_size=1 error while optimizing affine params
        for step in range(3):
            optimizer.zero_grad()
            out = model(x40, x100, x200, x400)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            p = F.softmax(logits, dim=1)
            # Shannon entropy
            ent = -(p * torch.log(p + 1e-7)).sum(dim=1).mean()
            # Confidence margin weighting (HAT-MIL)
            conf, _ = torch.max(p, dim=1)
            margin_weight = torch.clamp(conf - 0.5, min=0.1)
            # Marginal diversity regularization
            p_mean = p.mean(dim=0)
            div = (p_mean * torch.log(p_mean + 1e-7)).sum()
            loss = (margin_weight * ent).mean() + 0.1 * div
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x40, x100, x200, x400)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            hat_probs.append(F.softmax(logits, dim=1)[0, 1].item())

    m_hat = compute_classification_metrics(bag_labels, hat_probs)

    # Restore base state
    model.load_state_dict(base_state)

    tta_summary = {
        'Source': {'Accuracy': round(m_source['accuracy'], 4), 'AUROC': round(m_source['auroc'] or 0.5, 4), 'F1': round(m_source['f1_macro'], 4), 'ECE': round(m_source['ece'], 4)},
        'TENT': {'Accuracy': round(m_tent['accuracy'], 4), 'AUROC': round(m_tent['auroc'] or 0.5, 4), 'F1': round(m_tent['f1_macro'], 4), 'ECE': round(m_tent['ece'], 4)},
        'HAT-MIL': {'Accuracy': round(m_hat['accuracy'], 4), 'AUROC': round(m_hat['auroc'] or 0.5, 4), 'F1': round(m_hat['f1_macro'], 4), 'ECE': round(m_hat['ece'], 4)},
    }
    for k, v in tta_summary.items():
        print(f"  {k:<12} | Accuracy: {v['Accuracy']:.4f} | AUROC: {v['AUROC']:.4f} | F1: {v['F1']:.4f} | ECE: {v['ECE']:.4f}")
    return tta_summary

# ─────────────────────────────────────────────────────────────────────────────
# MASTER BENCHMARK EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    benchmark_results = {}

    print("\n" + "="*78)
    print(" 1. EVALUATION OF MAGNET (BEST_MODEL.PTH) ON FIXED TEST SET (13 PATIENTS)")
    print("="*78)
    best_model = MAGNET(pretrained=False).to(device)
    best_model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device), strict=True)
    m_test, preds_test, labs_test, probs_test = evaluate_model_on_patients(best_model, test_patients, batch_size=4, num_repeats=5)
    
    print(f"MAGNET Best Model Test Metrics (N={len(test_patients)} patients):")
    print(f"  Accuracy  : {m_test['accuracy']:.4f}")
    print(f"  AUROC     : {m_test['auroc']:.4f}")
    print(f"  Macro-F1  : {m_test['f1_macro']:.4f}")
    print(f"  Precision : {m_test['precision']:.4f}")
    print(f"  Recall    : {m_test['recall']:.4f}")
    print(f"  ECE       : {m_test['ece']:.4f}")
    print(f"  Brier     : {m_test['brier_score']:.4f}")
    benchmark_results['fixed_split_test'] = m_test

    print("\n" + "="*78)
    print(" 2. SYSTEMATIC ABLATION STUDY ON PHYSICAL TEST SET")
    print("="*78)
    ablation_configs = [
        ('MAGNET (Full)', 'checkpoints/best_model.pth', True, True, True),
        ('w/o GNN', 'checkpoints/ablation_wo_GNN.pth', False, True, True),
        ('w/o Aux Supervision', 'checkpoints/ablation_wo_Aux.pth', True, False, True),
        ('w/o Cross-Mag Attn', 'checkpoints/ablation_wo_Attn.pth', True, True, False),
        ('w/o Both (GNN + Aux)', 'checkpoints/ablation_wo_Both.pth', False, False, True),
    ]

    ablation_table = []
    for name, ckpt_path, use_gnn, use_aux, use_attn in ablation_configs:
        m = MAGNET(pretrained=False, use_gnn=use_gnn, use_aux=use_aux).to(device)
        if not use_attn:
            def mean_fwd(self, feats):
                x = torch.stack(feats, dim=1)
                return x.mean(1), torch.zeros(x.size(0), 4, 4, device=x.device)
            m.fusion.forward = mean_fwd.__get__(m.fusion, type(m.fusion))
        m.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
        m_ab, _, _, _ = evaluate_model_on_patients(m, test_patients, batch_size=4, num_repeats=5)
        row = {
            'Configuration': name,
            'Accuracy': round(m_ab['accuracy'], 4),
            'AUROC': round(m_ab['auroc'], 4) if m_ab['auroc'] is not None else 0.5,
            'Macro-F1': round(m_ab['f1_macro'], 4),
            'ECE': round(m_ab['ece'], 4),
            'Brier': round(m_ab['brier_score'], 4)
        }
        ablation_table.append(row)
        print(f"  {name:<24} | Acc: {row['Accuracy']:.4f} | AUROC: {row['AUROC']:.4f} | F1: {row['Macro-F1']:.4f} | ECE: {row['ECE']:.4f}")
    
    df_ablation = pd.DataFrame(ablation_table)
    df_ablation.to_csv('experiments/results/real_ablation_results.csv', index=False)
    benchmark_results['ablations'] = ablation_table

    print("\n" + "="*78)
    print(" 3. 5-FOLD CROSS-VALIDATION ACROSS ALL 82 PATIENTS")
    print("="*78)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs, fold_aurocs, fold_f1s, fold_eces = [], [], [], []
    oof_magnet_preds = [0] * len(all_patients)
    oof_magnet_probs = [0.0] * len(all_patients)

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_patients, pat_labels)):
        fold_patients = [all_patients[i] for i in val_idx]
        ckpt_path = f'checkpoints/fold_{fold+1}.pth'
        m_fold = MAGNET(pretrained=False).to(device)
        m_fold.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)

        m_f, p_f, _, pr_f = evaluate_model_on_patients(m_fold, fold_patients, batch_size=4, num_repeats=5)
        fold_accs.append(m_f['accuracy'])
        fold_aurocs.append(m_f['auroc'] if m_f['auroc'] is not None else 0.5)
        fold_f1s.append(m_f['f1_macro'])
        fold_eces.append(m_f['ece'])

        for idx, orig_i in enumerate(val_idx):
            oof_magnet_preds[orig_i] = p_f[idx]
            oof_magnet_probs[orig_i] = pr_f[idx]

        print(f"  Fold {fold+1} (N={len(fold_patients)}) | Acc: {m_f['accuracy']:.4f} | AUROC: {fold_aurocs[-1]:.4f} | F1: {m_f['f1_macro']:.4f} | ECE: {m_f['ece']:.4f}")

    cv_summary = {
        'mean_accuracy': round(float(np.mean(fold_accs)), 4),
        'ci95_accuracy': round(compute_t_ci(fold_accs), 4),
        'mean_auroc': round(float(np.mean(fold_aurocs)), 4),
        'ci95_auroc': round(compute_t_ci(fold_aurocs), 4),
        'mean_f1': round(float(np.mean(fold_f1s)), 4),
        'ci95_f1': round(compute_t_ci(fold_f1s), 4),
        'mean_ece': round(float(np.mean(fold_eces)), 4),
        'ci95_ece': round(compute_t_ci(fold_eces), 4),
        'fold_accuracies': [round(x, 4) for x in fold_accs],
        'fold_aurocs': [round(x, 4) for x in fold_aurocs],
        'fold_f1s': [round(x, 4) for x in fold_f1s],
    }
    print(f"\nMAGNET 5-Fold CV Summary (N=82 Patients):")
    print(f"  Mean Accuracy : {cv_summary['mean_accuracy']:.4f} ± {cv_summary['ci95_accuracy']:.4f}")
    print(f"  Mean AUROC    : {cv_summary['mean_auroc']:.4f} ± {cv_summary['ci95_auroc']:.4f}")
    print(f"  Mean Macro-F1 : {cv_summary['mean_f1']:.4f} ± {cv_summary['ci95_f1']:.4f}")
    print(f"  Mean ECE      : {cv_summary['mean_ece']:.4f} ± {cv_summary['ci95_ece']:.4f}")
    benchmark_results['5fold_cv'] = cv_summary

    print("\n" + "="*78)
    print(" 4. EVALUATION OF BASELINES & STATISTICAL SIGNIFICANCE TESTING")
    print("="*78)
    oof_baseline_preds = [0] * len(all_patients)
    # Evaluate 5-fold CV baseline models (baseline_cv_fold_*.pth: EfficientNet-B0 @ 40X)
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_patients, pat_labels)):
        fold_patients = [all_patients[i] for i in val_idx]
        ckpt_path = f'checkpoints/baseline_cv_fold_{fold+1}.pth'
        b_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2).to(device)
        b_model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
        b_model.eval()

        for idx, orig_i in enumerate(val_idx):
            pat_id = all_patients[orig_i]
            paths = patient_info[pat_id]['paths']['40X']
            if paths:
                tile_probs = []
                with torch.no_grad():
                    for p in paths[:5]:
                        img = Image.open(p).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                        img = macenko_norm(img)
                        t = val_tf(img).unsqueeze(0).to(device)
                        lp = F.softmax(b_model(t), dim=1)[0, 1].item()
                        tile_probs.append(lp)
                b_prob = np.mean(tile_probs)
                b_pred = int(b_prob >= 0.5)
            else:
                b_pred = 0
            oof_baseline_preds[orig_i] = b_pred

    base_acc = accuracy_score(pat_labels, oof_baseline_preds)
    base_f1 = f1_score(pat_labels, oof_baseline_preds, average='macro')
    print(f"Baseline (EfficientNet-B0 @ 40X) 5-Fold CV Accuracy: {base_acc:.4f}, Macro-F1: {base_f1:.4f}")

    mcnemar = paired_mcnemar_test(pat_labels, oof_magnet_preds, oof_baseline_preds)
    print("\nEdwards' Paired McNemar Test (with continuity correction):")
    print(f"  b (MAGNET correct, Baseline wrong) : {mcnemar['b']}")
    print(f"  c (Baseline correct, MAGNET wrong) : {mcnemar['c']}")
    print(f"  Chi-Square Statistic               : {mcnemar['chi2']:.4f}")
    print(f"  p-value                            : {mcnemar['p_value']:.4e} (Significant: {mcnemar['significant']})")
    benchmark_results['mcnemar_test'] = mcnemar
    benchmark_results['baseline_cv'] = {'accuracy': round(base_acc, 4), 'macro_f1': round(base_f1, 4)}

    # Evaluate Concat baseline on fixed test split
    concat_model = ConcatFusionBaseline().to(device)
    concat_model.load_state_dict(torch.load('checkpoints/concat_baseline.pth', map_location=device), strict=True)
    m_concat, _, _, _ = evaluate_model_on_patients(concat_model, test_patients, batch_size=4, num_repeats=5)
    print(f"\nConcatFusion Baseline Test Accuracy: {m_concat['accuracy']:.4f}, AUROC: {m_concat['auroc']:.4f}, F1: {m_concat['f1_macro']:.4f}")
    benchmark_results['concat_baseline_test'] = m_concat

    # Run TTA physical evaluation
    tta_results = run_physical_tta_evaluation(best_model, test_patients)
    benchmark_results['tta_benchmark'] = tta_results

    # Save all real benchmark results
    with open('experiments/results/real_benchmark_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    print("\n" + "="*78)
    print(" ALL PHYSICAL BENCHMARK RESULTS SUCCESSFULLY SAVED TO:")
    print(" experiments/results/real_benchmark_results.json")
    print("="*78, flush=True)

if __name__ == '__main__':
    main()
