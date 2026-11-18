# Reproduction Guide & Execution Protocols

This repository is engineered for **100% mathematical and empirical reproducibility**. Follow the protocols below to set up the environment, verify datasets, reproduce benchmarks, and generate publication figures.

---

## 1. Quickstart Reproduction Steps

### Step 1: Clone Repository & Create Environment
```bash
git clone https://github.com/HippomasAKiB1/BreastHistography.git
cd BreastHistography

conda create -n breasthisto python=3.10 -y
conda activate breasthisto
pip install -r requirements.txt
```

### Step 2: Run Automated Unit Tests (Verify Installation)
```bash
pytest tests/ -v
```
*Expected: 14/14 tests passing in under 10 seconds.*

### Step 3: Dataset Ingestion & Verification
Ensure the BreaKHis dataset is placed under `data/BreaKHis_v1/`. Then run:
```bash
python scripts/verify_dataset.py
```
*Expected output: 82 patients (24 benign, 58 malignant) and 7,909 images verified.*

### Step 4: Reproduce Model Benchmark
```bash
python scripts/evaluate_checkpoints.py --model magnet --checkpoint checkpoints/best_model.pth
```

### Step 5: Reproduce TTA Benchmark
```bash
python scripts/run_tta_adaptation.py --output results/tta_benchmark_reproduced.json
```

### Step 6: Regenerate Publication Figures
```bash
python scripts/generate_figures.py
```
*Outputs: 300 DPI vector PDFs in `paper/figures/` and `plots/`.*

### Step 7: Package Manuscript Bundle
```bash
python scripts/package_manuscript.py
```
*Outputs: `paper/breasthistography_manuscript.zip` ready for 1-click import into Overleaf.*

---

## 2. Seed Control & Determinism

All training, cross-validation splitting, and evaluation routines use explicit deterministic seed controls:
- **Global Seed**: `42`
- **PyTorch Determinism**: `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`
- **Monte Carlo Multi-Pass Sampling**: Seeds computed as `1000 + pass_idx * 17` for exact evaluation repeatability across machines.

---

## 3. Hardware Requirements

| Configuration | Minimal (Inference / Testing) | Recommended (Full Training) |
| :--- | :--- | :--- |
| **CPU** | 4-Core x86_64 CPU | 8-Core Intel Core i7 / AMD Ryzen 7 |
| **RAM** | 8 GB System Memory | 16 GB System Memory |
| **GPU** | NVIDIA GTX 1650 (4 GB VRAM) | NVIDIA RTX 3080 / 4090 / A100 |
| **Disk Space** | ~500 MB (Code & Docs) | ~8 GB (with BreaKHis v1 dataset) |
