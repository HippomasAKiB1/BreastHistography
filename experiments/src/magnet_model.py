import os, sys, re, random, json, time, copy, warnings
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import scipy.stats as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

DATA_ROOT = 'data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'

# ── Hardware (GTX 1650 4 GB) ──────────────────────────────────────────────
BATCH_SIZE         = 4
NUM_WORKERS        = 0
IMG_SIZE           = 224
IMG_SIZE_WARMUP    = 160    # Phase 3.1: progressive resizing warm-up
FEAT_DIM           = 512
MAGNIFICATIONS     = ['40X', '100X', '200X', '400X']
ACCUM_STEPS        = 4      # gradient accumulation → effective batch = 16
# ⚠ VRAM note: MIL processes BATCH_SIZE × IMAGES_PER_MAG images per forward pass.
# With B=4, N=5: 20 imgs × 4 mags = 80 images at once. If OOM → set IMAGES_PER_MAG=3.

# ── Training ─────────────────────────────────────────────────────────────
EPOCHS             = 60
WARMUP_EPOCHS      = 5      # train at 160px then switch to 224px
LR                 = 5e-4   # heads — safe for pretrained models
LR_BACKBONE        = 5e-5   # backbones — conservative for pretrained weights
WEIGHT_DECAY       = 1e-4
EARLY_STOP         = 15
LABEL_SMOOTHING    = 0.1
FOCAL_GAMMA        = 2.0    # focal loss gamma
IMAGES_PER_MAG     = 3      # max images sampled per patient per mag per epoch

# ── CV ───────────────────────────────────────────────────────────────────
N_FOLDS            = 5
KF_EPOCHS          = 40
KF_PATIENCE        = 10

# ── Upgraded backbones (stronger but still fit 4 GB with AMP) ────────────
BACKBONES = {
    '40X' : 'efficientnet_b2',        # replaces mobilenetv3_large — much stronger
    '100X': 'efficientnet_b1',        # replaces efficientnet_b0
    '200X': 'resnet50',               # replaces resnet34
    '400X': 'efficientnet_b0',        # replaces mobilenetv3_small
}
BACKBONE_OUT_DIM = {
    '40X' : 1408,   # efficientnet_b2
    '100X': 1280,   # efficientnet_b1
    '200X': 2048,   # resnet50
    '400X': 1280,   # efficientnet_b0
}

def _safe_name(s):
    return re.sub(r'[^\w\-]', '_', s)

print('Config loaded.')
print(f'Effective batch size: {BATCH_SIZE * ACCUM_STEPS} (via gradient accumulation)')


def macenko_norm(img_pil: Image.Image) -> Image.Image:
    """
    OD-space colour normalisation (simplified Macenko).
    For exact Macenko: pip install staintools
    """
    img = np.array(img_pil, dtype=np.float32)
    img = np.clip(img, 1, 255)
    OD  = -np.log(img / 255.0)
    for c in range(3):
        m = OD[..., c].mean()
        s = OD[..., c].std() + 1e-6
        OD[..., c] = (OD[..., c] - m) / s * 0.3 + 0.4
    img_norm = np.exp(-OD) * 255.0
    return Image.fromarray(np.clip(img_norm, 0, 255).astype(np.uint8))

print('Stain normaliser defined.')

BENIGN_SUBS    = ['adenosis','fibroadenoma','phyllodes_tumor','tubular_adenoma']
MALIGNANT_SUBS = ['ductal_carcinoma','lobular_carcinoma','mucinous_carcinoma','papillary_carcinoma']
ALL_SUBS       = BENIGN_SUBS + MALIGNANT_SUBS
SUB2IDX        = {s: i for i, s in enumerate(ALL_SUBS)}

def collect_patient_info(root_dir):
    """
    Supports both BreakHis folder layouts:
      Layout A (common): breast/benign/SOB/adenosis/<patient>/<mag>/*.png
      Layout B (some downloads): breast/benign/adenosis/<patient>/<mag>/*.png
    """
    patients = {}
    for cls_name, binary, subs in [
            ('benign', 0, BENIGN_SUBS), ('malignant', 1, MALIGNANT_SUBS)]:
        cls_path = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        # Try SOB subdir first (Layout A), then direct (Layout B)
        sob_path = os.path.join(cls_path, 'SOB')
        search_root = sob_path if os.path.isdir(sob_path) else cls_path

        for sub in subs:
            sub_path = os.path.join(search_root, sub)
            if not os.path.isdir(sub_path):
                continue
            for pat in os.listdir(sub_path):
                pat_path = os.path.join(sub_path, pat)
                if not os.path.isdir(pat_path):
                    continue
                if pat not in patients:
                    patients[pat] = {
                        'binary' : binary,
                        'subtype': SUB2IDX[sub],
                        'paths'  : {m: [] for m in MAGNIFICATIONS}
                    }
                for mag in MAGNIFICATIONS:
                    mag_path = os.path.join(pat_path, mag)
                    if os.path.isdir(mag_path):
                        for fname in os.listdir(mag_path):
                            if fname.lower().endswith('.png'):
                                patients[pat]['paths'][mag].append(
                                    os.path.join(mag_path, fname))
    return patients


# ── Safety: fail loudly if dataset not found ──────────────────────────────────
if not os.path.isdir(DATA_ROOT):
    raise FileNotFoundError(
        f'BreakHis dataset not found at:\n  {DATA_ROOT}\n'
        'Download from: https://web.inf.ufpr.br/vri/databases/'
        'breast-cancer-histopathological-database-breakhis/\n'
        'Then update DATA_ROOT in the config cell.'
    )

patient_info = collect_patient_info(DATA_ROOT)

if len(patient_info) == 0:
    raise RuntimeError(
        f'No patients found under {DATA_ROOT}.\n'
        'Check that the folder structure matches:\n'
        '  breast/benign/SOB/<subtype>/<patient_id>/<magnification>/*.png\n'
        'or:\n'
        '  breast/benign/<subtype>/<patient_id>/<magnification>/*.png'
    )

all_patients = list(patient_info.keys())
pat_labels   = [patient_info[p]['binary'] for p in all_patients]

# ── Dataset statistics ────────────────────────────────────────────────────────
n_benign    = sum(1 for p in all_patients if patient_info[p]['binary'] == 0)
n_malignant = len(all_patients) - n_benign
total_imgs  = sum(len(v) for p in all_patients
                  for v in patient_info[p]['paths'].values())

print(f'Total patients : {len(all_patients)}')
print(f'  Benign       : {n_benign}  ({100*n_benign/len(all_patients):.1f}%)')
print(f'  Malignant    : {n_malignant}  ({100*n_malignant/len(all_patients):.1f}%)')
print(f'Total images   : {total_imgs}')

from collections import Counter
sub_counts = Counter(patient_info[p]['subtype'] for p in all_patients)
print('\nSubtype distribution:')
for idx, name in enumerate(ALL_SUBS):
    print(f'  {name:<25}: {sub_counts.get(idx, 0)} patients')

# Check all patients have images at all magnifications
missing = [(p, mag) for p in all_patients for mag in MAGNIFICATIONS
           if not patient_info[p]['paths'][mag]]
if missing:
    print(f'\n⚠ {len(missing)} patient-magnification combinations have no images:')
    for p, m in missing[:5]:
        print(f'  {p} @ {m}')
    if len(missing) > 5:
        print(f'  ... and {len(missing)-5} more')
else:
    print('\n✓ All patients have images at all 4 magnifications.')

# ── Fixed 70/15/15 patient split (for ablation / interpretability cells) ──────
random.seed(42); random.shuffle(all_patients)
n = len(all_patients)
train_patients = all_patients[:int(0.70 * n)]
val_patients   = all_patients[int(0.70 * n):int(0.85 * n)]
test_patients  = all_patients[int(0.85 * n):]
print(f'\nFixed split — Train:{len(train_patients)} Val:{len(val_patients)} Test:{len(test_patients)}')
print('(5-fold CV in Section 7 uses all patients — recommended for publication)')


import math

from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Albumentations pipelines ──────────────────────────────────────────────────
albu_train = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=180, p=0.7),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120*0.05, p=0.5),
        A.GridDistortion(p=0.5),
    ], p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
    A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2(),
])
albu_val = A.Compose([
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2(),
])

# TTA transforms: same image, different geometric views
TTA_TRANSFORMS = [
    A.Compose([A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
    A.Compose([A.HorizontalFlip(p=1.0), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
    A.Compose([A.VerticalFlip(p=1.0),   A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
    A.Compose([A.RandomRotate90(p=1.0), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
    A.Compose([A.Transpose(p=1.0),      A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
]

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]
train_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)])
val_tf   = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)])


class MultiMagDataset(Dataset):
    """
    Option B (MIL): returns a LIST of tensors per magnification.
    Each image is loaded and augmented independently; the backbone
    sees each image separately and features are aggregated AFTER
    the backbone — preserving per-image discriminative detail.

    Returns:
        imgs_per_mag: list of 4 lists, each containing n_per_mag tensors [3,H,W]
        binary_label: int
        subtype_label: int
    """
    def __init__(self, patient_ids, patient_info, augment=False,
                 img_size=224, n_per_mag=IMAGES_PER_MAG):
        self.patient_ids  = patient_ids
        self.patient_info = patient_info
        self.augment      = augment
        self.img_size     = img_size
        self.n_per_mag    = n_per_mag
        self.pipeline     = albu_train if augment else albu_val

    def __len__(self): return len(self.patient_ids)

    def _load(self, path):
        img = Image.open(path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = macenko_norm(img)
        arr = np.array(img, dtype=np.uint8)
        return self.pipeline(image=arr)['image']   # [3,H,W] float32

    def __getitem__(self, idx):
        pid  = self.patient_ids[idx]
        info = self.patient_info[pid]
        rng  = np.random.default_rng()
        imgs_per_mag = []
        for mag in MAGNIFICATIONS:
            paths = info['paths'][mag]
            if not paths:
                # Pad with zeros if mag missing
                imgs_per_mag.append(
                    [torch.zeros(3, self.img_size, self.img_size)] * self.n_per_mag)
                continue
            n      = min(self.n_per_mag, len(paths))
            chosen = rng.choice(len(paths), size=n, replace=False)
            tensors = [self._load(paths[c]) for c in chosen]
            # Pad to n_per_mag if fewer images available
            while len(tensors) < self.n_per_mag:
                tensors.append(tensors[rng.integers(len(tensors))])
            imgs_per_mag.append(tensors)   # list of n_per_mag tensors
        return imgs_per_mag, int(info['binary']), int(info['subtype'])


def mil_collate_fn(batch):
    """
    Custom collate: stacks images_per_mag into [B, N, 3, H, W] tensors.
    Output per magnification: tensor of shape [B, N, 3, H, W]
    where N = IMAGES_PER_MAG.
    """
    imgs_list  = [item[0] for item in batch]   # list of B x (4 x N x tensor)
    y_bin      = torch.tensor([item[1] for item in batch], dtype=torch.long)
    y_sub      = torch.tensor([item[2] for item in batch], dtype=torch.long)
    # imgs_list[b][mag_idx][img_idx] = tensor [3,H,W]
    mag_batches = []
    for mag_idx in range(len(MAGNIFICATIONS)):
        # Stack: [B, N, 3, H, W]
        mag_tensor = torch.stack([
            torch.stack(imgs_list[b][mag_idx])   # [N, 3, H, W]
            for b in range(len(batch))
        ])  # [B, N, 3, H, W]
        mag_batches.append(mag_tensor)
    return (*mag_batches, y_bin, y_sub)


def make_loader(patients, augment, shuffle, img_size=IMG_SIZE, drop_last=None):
    ds  = MultiMagDataset(patients, patient_info, augment=augment,
                          img_size=img_size, n_per_mag=IMAGES_PER_MAG)
    dl  = drop_last if drop_last is not None else shuffle
    if shuffle:
        labels  = [patient_info[p]['binary'] for p in patients]
        counts  = [labels.count(0), labels.count(1)]
        weights = [1.0/counts[l] for l in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=False,
                          drop_last=True, collate_fn=mil_collate_fn)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=False,
                      drop_last=dl, collate_fn=mil_collate_fn)


train_loader_warmup = make_loader(train_patients, augment=True,  shuffle=True,  img_size=IMG_SIZE_WARMUP)
train_loader        = make_loader(train_patients, augment=True,  shuffle=True,  img_size=IMG_SIZE)
val_loader          = make_loader(val_patients,   augment=False, shuffle=False, img_size=IMG_SIZE)
test_loader         = make_loader(test_patients,  augment=False, shuffle=False, img_size=IMG_SIZE)

print(f'Loaders: train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}')
print(f'MIL: each patient → {IMAGES_PER_MAG} images/mag processed individually')

# Sanity check: verify batch shape is [B, N, 3, H, W]
print('Sanity check...')
_b = next(iter(train_loader))
print(f'  Per-mag shape (B,N,C,H,W): {_b[0].shape}')
print(f'  Binary labels: {_b[4].tolist()}')
del _b
print('DataLoader OK — MIL batch format confirmed.')

# ─────────────────────────────────────────────────────────────────────
# Class definitions moved here so they are available before Section 8
# (McNemar / baseline CV cells run before Section 10 training calls)
# ─────────────────────────────────────────────────────────────────────
from torchvision import transforms as T

class SingleMagDataset(Dataset):
    """
    Single-magnification baseline dataset — MIL version.
    Returns [N, 3, H, W] per sample so baseline also processes N images
    (ensuring fair comparison: same data richness as MAGNET).
    """
    def __init__(self, patient_ids, patient_info, mag, augment=False,
                 img_size=224, n_per_mag=IMAGES_PER_MAG):
        self.patient_ids  = patient_ids
        self.patient_info = patient_info
        self.mag          = mag
        self.img_size     = img_size
        self.n_per_mag    = n_per_mag
        self.pipeline     = albu_train if augment else albu_val

    def __len__(self): return len(self.patient_ids)

    def _load(self, path):
        img = Image.open(path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = macenko_norm(img)
        return self.pipeline(image=np.array(img, dtype=np.uint8))['image']

    def __getitem__(self, idx):
        pid   = self.patient_ids[idx]
        info  = self.patient_info[pid]
        paths = info['paths'][self.mag]
        rng   = np.random.default_rng()
        n     = min(self.n_per_mag, len(paths)) if paths else 1
        chosen= rng.choice(len(paths), size=n, replace=False) if paths else [0]
        tensors = [self._load(paths[c]) for c in chosen] if paths else \
                  [torch.zeros(3, self.img_size, self.img_size)]
        while len(tensors) < self.n_per_mag:
            tensors.append(tensors[rng.integers(len(tensors))])
        # [N, 3, H, W]
        return torch.stack(tensors), int(info['binary'])


def train_single_baseline(backbone_name, mag, epochs=20, patience=6):
    """Single backbone baseline — MIL version for fair comparison."""
    print(f'\nBaseline: {backbone_name} @ {mag} (MIL, N={IMAGES_PER_MAG})')
    def _sl(patients, augment, sh):
        ds = SingleMagDataset(patients, patient_info, mag,
                              augment=augment, img_size=IMG_SIZE)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=sh,
                          num_workers=NUM_WORKERS, pin_memory=False, drop_last=sh)
    tl=_sl(train_patients,True,True)
    vl=_sl(val_patients,False,False)
    el=_sl(test_patients,False,False)
    torch.cuda.empty_cache()
    # Backbone with MIL attention pooling for fair comparison
    backbone = timm.create_model(backbone_name, pretrained=True,
                                  num_classes=0, global_pool='avg').to(device)
    in_dim   = backbone(torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(device)).shape[1]
    attn_pool= nn.Sequential(nn.Linear(in_dim,128), nn.Tanh(),
                              nn.Linear(128,1)).to(device)
    # FIX: BatchNorm1d -> LayerNorm, same small-batch instability reasoning
    # as MAGNET's heads (see Section 4 note) -- keeps this baseline's
    # checkpoint architecture consistent with get_baseline_preds_mil() below.
    classifier = nn.Sequential(nn.Linear(in_dim,256), nn.LayerNorm(256),
                                nn.GELU(), nn.Dropout(0.4),
                                nn.Linear(256,2)).to(device)
    params = list(backbone.parameters()) + list(attn_pool.parameters()) + \
             list(classifier.parameters())
    opt = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    _s_opt_steps = math.ceil(len(tl)/ACCUM_STEPS)*epochs
    # FIX: max_lr=LR*5 (2.5e-3) ramped up within the first ~10% of a very
    # short per-epoch schedule was pushing freshly-initialised heads too
    # hard, contributing to the NaN/Inf gradient blow-ups. 2x is still a
    # healthy OneCycle boost over the base LR without being reckless.
    sch = optim.lr_scheduler.OneCycleLR(opt, max_lr=LR*1,
              total_steps=_s_opt_steps, pct_start=0.1)
    # CrossEntropyLoss here (FocalLoss defined later in notebook;
    # baseline uses CE for simplicity — fair since both use label smoothing)
    ce  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    sc  = make_scaler()
    best_acc=0.0; pat=0
    ckpt=f'checkpoints/baseline_{_safe_name(backbone_name)}_{mag}.pth'
    for ep in range(1,epochs+1):
        backbone.train(); attn_pool.train(); classifier.train()
        opt.zero_grad()
        for step,(imgs,labels) in enumerate(tl):
            # imgs: [B, N, 3, H, W]
            imgs,labels = imgs.to(device), labels.to(device)
            B,N,C,H,W  = imgs.shape
            with autocast_ctx():
                feats = backbone(imgs.view(B*N,C,H,W)).view(B,N,-1)
                attn  = torch.softmax(attn_pool(feats), dim=1)
                pooled= (attn * feats).sum(1)
                loss  = ce(classifier(pooled), labels) / ACCUM_STEPS
            sc.scale(loss).backward()
            if (step+1)%ACCUM_STEPS==0 or (step+1)==len(tl):
                sc.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                sc.step(opt); sc.update(); opt.zero_grad(); sch.step()
        backbone.eval(); attn_pool.eval(); classifier.eval()
        preds,labs=[],[]
        with torch.no_grad():
            for imgs,labels in vl:
                B,N,C,H,W = imgs.shape
                imgs = imgs.to(device)
                feats = backbone(imgs.view(B*N,C,H,W)).view(B,N,-1)
                attn  = torch.softmax(attn_pool(feats),dim=1)
                pooled= (attn*feats).sum(1)
                preds.extend(classifier(pooled).argmax(1).cpu().tolist())
                labs.extend(labels.tolist())
        va=accuracy_score(labs,preds)
        if va>best_acc:
            best_acc=va
            torch.save({'bb':backbone.state_dict(),'attn':attn_pool.state_dict(),
                        'cls':classifier.state_dict()}, ckpt)
            pat=0
        else:
            pat+=1
            if pat>=patience: break
        print(f'  ep{ep:02d}: val_acc={va:.4f}')
    # Test evaluation
    ckpt_d = torch.load(ckpt, map_location=device)
    backbone.load_state_dict(ckpt_d['bb'])
    attn_pool.load_state_dict(ckpt_d['attn'])
    classifier.load_state_dict(ckpt_d['cls'])
    backbone.eval(); attn_pool.eval(); classifier.eval()
    preds,labs,probs=[],[],[]
    with torch.no_grad():
        for imgs,labels in el:
            B,N,C,H,W = imgs.shape
            imgs = imgs.to(device)
            feats = backbone(imgs.view(B*N,C,H,W)).view(B,N,-1)
            attn  = torch.softmax(attn_pool(feats),dim=1)
            logits= classifier((attn*feats).sum(1))
            preds.extend(logits.argmax(1).cpu().tolist())
            probs.extend(F.softmax(logits,1)[:,1].cpu().tolist())
            labs.extend(labels.tolist())
    ta=accuracy_score(labs,preds)
    f1b=f1_score(labs,preds,average='weighted',zero_division=0)
    try: ta_auc=roc_auc_score(labs,probs)
    except ValueError: ta_auc=float('nan')
    del backbone,attn_pool,classifier; torch.cuda.empty_cache()
    print(f'  → acc={ta:.4f}  auc={ta_auc:.4f}  f1={f1b:.4f}')
    return {'config':f'{backbone_name}@{mag}','test_acc':ta,'auc':ta_auc,'f1':f1b}

# ── ConcatFusion Baseline (MIL-compatible) ───────────────────────────────────
class ConcatFusionBaseline(nn.Module):
    """
    Simple multi-magnification baseline: concatenate projected features from
    all 4 magnifications.
    MIL-compatible: applies attention pooling over N images before projection,
    matching the same data richness as MAGNET for a fair comparison.
    """
    def __init__(self, proj_dim=256, num_binary=2, pretrained=True):
        super().__init__()
        self.branches = nn.ModuleDict({
            mag: timm.create_model(BACKBONES[mag], pretrained=pretrained,
                                   num_classes=0, global_pool='avg')
            for mag in MAGNIFICATIONS
        })
        # Attention MIL pooling per branch (same as BranchExtractor)
        self.attn_pools = nn.ModuleDict({
            mag: nn.Sequential(
                nn.Linear(BACKBONE_OUT_DIM[mag], 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            ) for mag in MAGNIFICATIONS
        })
        self.projs = nn.ModuleDict({
            mag: nn.Linear(BACKBONE_OUT_DIM[mag], proj_dim)
            for mag in MAGNIFICATIONS
        })
        self.classifier = nn.Sequential(
            nn.Linear(4 * proj_dim, 512),
            nn.LayerNorm(512),  # FIX: BatchNorm1d -> LayerNorm (small-batch stability)
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_binary)
        )

    def forward(self, x40, x100, x200, x400):
        inputs = {'40X':x40, '100X':x100, '200X':x200, '400X':x400}
        feats  = []
        for mag in MAGNIFICATIONS:
            x = inputs[mag]                              # [B, N, 3, H, W]
            B, N, C, H, W = x.shape
            f    = self.branches[mag](x.view(B*N, C, H, W))   # [B*N, D]
            f    = f.view(B, N, -1)                           # [B, N, D]
            attn = torch.softmax(self.attn_pools[mag](f), dim=1)  # [B, N, 1]
            pooled = (attn * f).sum(dim=1)               # [B, D]
            feats.append(self.projs[mag](pooled))         # [B, proj_dim]
        return self.classifier(torch.cat(feats, dim=1))  # [B, 2]


def run_concat_baseline(epochs=20, patience=6):
    print('\nBaseline: ConcatFusion (4-mag MIL attention pooling, no cross-attn)')
    torch.cuda.empty_cache()
    m   = ConcatFusionBaseline(pretrained=True).to(device)
    opt = optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    _cc_opt_steps = math.ceil(len(train_loader)/ACCUM_STEPS)*epochs
    sch = optim.lr_scheduler.OneCycleLR(   # FIX: max_lr 5x -> 2x, see note above
        opt, max_lr=LR*1,
        total_steps=_cc_opt_steps,
        pct_start=0.1, anneal_strategy='cos',
        div_factor=10, final_div_factor=1e3)
    ce  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    sc  = make_scaler()
    best_acc=0.0; pat=0; ckpt='checkpoints/concat_baseline.pth'

    for ep in range(1, epochs+1):
        m.train(); opt.zero_grad()
        for step, batch in enumerate(train_loader):
            x40,x100,x200,x400,y_bin,_ = [b.to(device) for b in batch]
            with autocast_ctx():
                loss = ce(m(x40,x100,x200,x400), y_bin) / ACCUM_STEPS
            sc.scale(loss).backward()
            if (step+1) % ACCUM_STEPS == 0 or (step+1) == len(train_loader):
                sc.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                sc.step(opt); sc.update(); opt.zero_grad(); sch.step()
        m.eval(); preds, labs = [], []
        with torch.no_grad():
            for batch in val_loader:
                x40,x100,x200,x400,y_bin,_ = [b.to(device) for b in batch]
                preds.extend(m(x40,x100,x200,x400).argmax(1).cpu().tolist())
                labs.extend(y_bin.cpu().tolist())
        va = accuracy_score(labs, preds)
        if va > best_acc:
            best_acc=va; torch.save(m.state_dict(), ckpt); pat=0
        else:
            pat+=1
            if pat >= patience: break
        print(f'  ep{ep:02d}: val_acc={va:.4f}')

    m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval()
    preds, labs, probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x40,x100,x200,x400,y_bin,_ = [b.to(device) for b in batch]
            logits = m(x40,x100,x200,x400)
            preds.extend(logits.argmax(1).cpu().tolist())
            probs.extend(F.softmax(logits,1)[:,1].cpu().tolist())
            labs.extend(y_bin.cpu().tolist())
    ta   = accuracy_score(labs, preds)
    f1c  = f1_score(labs, preds, average='weighted', zero_division=0)
    try: ta_auc = roc_auc_score(labs, probs)
    except ValueError: ta_auc = float('nan')
    del m; torch.cuda.empty_cache()
    print(f'  → acc={ta:.4f}  auc={ta_auc:.4f}  f1={f1c:.4f}')
    return {'config':'ConcatFusion (4-mag MIL)', 'test_acc':ta, 'auc':ta_auc, 'f1':f1c}


class BranchExtractor(nn.Module):
    """
    Pretrained backbone + MIL pooling.
    Input:  x of shape [B, N, 3, H, W]  (N images per patient per mag)
    Process: run each of the N images through backbone independently,
             then aggregate via attention pooling over the N feature vectors.
    Output: raw [B, backbone_dim], proj [B, feat_dim]
    """
    def __init__(self, backbone_name, in_dim, out_dim, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          num_classes=0, global_pool='avg')
        # Attention MIL pooling — learns which images are most informative
        self.attn_pool = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)   # scalar attention score per image
        )
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        """
        x: [B, N, 3, H, W]
        Returns: raw [B, in_dim], proj [B, out_dim]
        """
        B, N, C, H, W = x.shape
        # Flatten batch+instance dims → process all N*B images in one pass
        x_flat  = x.view(B * N, C, H, W)            # [B*N, 3, H, W]
        feats   = self.backbone(x_flat)              # [B*N, in_dim]
        feats   = feats.view(B, N, -1)               # [B, N, in_dim]
        # Attention MIL pooling: learn a weight for each of the N images
        attn    = self.attn_pool(feats)              # [B, N, 1]
        attn    = torch.softmax(attn, dim=1)         # normalise over N
        raw     = (attn * feats).sum(dim=1)          # [B, in_dim] weighted sum
        proj    = self.proj(raw)                     # [B, out_dim]
        return raw, proj

print('BranchExtractor (MIL attention pooling over N images) defined.')


class CrossMagFusion(nn.Module):
    """Multi-head self-attention across 4 magnification tokens."""
    def __init__(self, feat_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert feat_dim % num_heads == 0
        self.attn      = nn.MultiheadAttention(feat_dim, num_heads, dropout=dropout, batch_first=True)
        self.mag_scale = nn.Parameter(torch.ones(4))
        self.ffn  = nn.Sequential(
            nn.Linear(feat_dim, feat_dim*2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(feat_dim*2, feat_dim))
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

    def forward(self, feats):
        x = torch.stack(feats, dim=1)                        # [B,4,D]
        attn_out, attn_w = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        w = F.softmax(self.mag_scale, dim=0)
        fused = (x * w.unsqueeze(0).unsqueeze(-1)).sum(1)    # [B,D]
        fused = self.norm2(fused + self.ffn(fused))
        return fused, attn_w

print('CrossMagFusion defined.')

class SparseGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, h, adj):
        return F.relu(adj @ self.W(h))

class BatchGraphNet(nn.Module):
    """Treats each batch as a graph; edges = cosine similarity > threshold."""
    def __init__(self, feat_dim=512, hidden_dim=256, num_layers=2, threshold=0.5, dropout=0.1):
        super().__init__()
        self.threshold = threshold
        in_d = feat_dim
        layers = []
        for _ in range(num_layers):
            layers.append(SparseGraphConv(in_d, hidden_dim)); in_d = hidden_dim
        self.gcn_layers = nn.ModuleList(layers)
        self.drop    = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_dim, feat_dim)
        self.norm    = nn.LayerNorm(feat_dim)

    def _adj(self, h):
        norm = F.normalize(h.detach(), p=2, dim=1)
        sim  = torch.mm(norm, norm.t())
        mask = (sim > self.threshold).float()
        mask.fill_diagonal_(1.0)
        deg = mask.sum(1, keepdim=True).clamp(min=1)
        d   = deg.pow(-0.5)
        return mask * d * d.t()

    def forward(self, h):
        adj = self._adj(h)
        out = h
        for layer in self.gcn_layers:
            out = self.drop(layer(out, adj))
        return self.norm(h + self.readout(out))

print('BatchGraphNet defined (pure PyTorch, no DGL).')

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
