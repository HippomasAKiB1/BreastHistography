# BreastHistography

An open-source research framework for multi-magnification breast cancer histopathology classification using cross-magnification attention fusion, patient graph networks, and head-aware test-time adaptation.

[![CI](https://github.com/HippomasAKiB1/BreastHistography/actions/workflows/ci.yml/badge.svg)](https://github.com/HippomasAKiB1/BreastHistography/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## Overview

In clinical breast cancer diagnostics, pathologists examine tissue specimens across multiple optical magnification scales (40X, 100X, 200X, 400X) to evaluate tissue architecture, glandular patterns, and cytological morphology.

BreastHistography implements:
1. **MAGNET**: A multi-scale attention network with specialized backbones for each optical magnification tier, fused via multihead cross-attention and inter-sample patient graph convolution.
2. **HAT-MIL**: Head-aware test-time adaptation with prototype alignment, preventing single-class collapse on imbalanced clinical distributions.
3. **Leakage-Proof Evaluation**: Patient-stratified 5-fold cross-validation ensuring zero overlap of patient slides between splits.

---

## Repository Structure

```
BreastHistography/
├── configs/               # Hyperparameter and experiment configurations
├── docs/                  # Detailed research, methodology, and dataset docs
│   ├── RESEARCH_OVERVIEW.md
│   ├── METHODOLOGY.md
│   ├── DATASET.md
│   ├── EXPERIMENTS.md
│   ├── RESULTS.md
│   ├── REPRODUCIBILITY.md
│   └── ARTIFACTS.md
├── paper/                 # Manuscript LaTeX files, figures, and Overleaf bundle
│   ├── main.tex
│   ├── references.bib
│   ├── figures/
│   └── breasthistography_manuscript.zip
├── results/               # Empirical benchmark outputs and manifests
├── scripts/               # CLI utilities (verification, evaluation, figures)
│   ├── verify_dataset.py
│   ├── download_breakhis.py
│   ├── evaluate_checkpoints.py
│   ├── run_tta_adaptation.py
│   ├── generate_figures.py
│   └── package_manuscript.py
├── src/                   # Core modular Python package
│   ├── models/            # MAGNET, backbones, fusion, graph network, baselines
│   ├── data/              # Macenko stain normalization, transforms, splits
│   ├── tta/               # TENT, HAT-MIL, adaptation adapter
│   ├── eval/              # Metrics, calibration (ECE), McNemar test
│   └── utils/             # I/O, seeding, logging
├── tests/                 # Automated PyTest suite (14/14 passing)
├── CITATION.cff           # Citation metadata
├── Makefile               # Development automation targets
├── pyproject.toml         # Package build specification
└── requirements.txt       # Dependencies
```

---

## Quickstart

### Installation
```bash
git clone https://github.com/HippomasAKiB1/BreastHistography.git
cd BreastHistography
pip install -r requirements.txt
```

### Run Tests
```bash
pytest tests/ -v
```

### Verify Dataset
```bash
python scripts/verify_dataset.py --data_dir data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast
```

### Benchmark Evaluation
```bash
python scripts/evaluate_checkpoints.py --model magnet --checkpoint checkpoints/best_model.pth
```

### Run TTA Benchmark
```bash
python scripts/run_tta_adaptation.py
```

### Generate Figures
```bash
python scripts/generate_figures.py
```

---

## Benchmark Results

Evaluated on the BreaKHis cohort using patient-stratified evaluation:

| Model Architecture | Magnification | Accuracy | Macro-F1 | AUROC | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| MobileNetV3-Large | 40X | 84.6% | 0.822 | 0.861 | 0.221 |
| EfficientNet-B0 | 100X | 84.6% | 0.822 | 0.889 | 0.198 |
| ResNet-34 | 200X | 84.6% | 0.822 | 0.917 | 0.205 |
| MobileNetV3-Small | 400X | 76.9% | 0.732 | 0.833 | 0.252 |
| Concat-Fusion Baseline | Multi | 92.3% | 0.909 | 0.972 | 0.124 |
| **MAGNET (Full)** | **Multi** | **100.0%** | **1.000** | **1.000** | **0.042** |

---

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{pyil2027breasthistography,
  author    = {Akib Hasan Pyil and Mohammad Rabib Uddin},
  title     = {Multi-Scale Attention-based Graph Networks with Head-Aware Test-Time Adaptation for Breast Cancer Histopathology},
  booktitle = {IEEE Engineering in Medicine and Biology Society (EMBC)},
  year      = {2027}
}
```

---

## License

This project is licensed under the MIT License.
