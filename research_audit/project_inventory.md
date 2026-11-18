# Project Inventory & Artifact Lineage Audit

**Project:** Diagnosing Test-Time Adaptation Failures in Multi-Instance Learning for Breast Cancer Histopathology / MAGNET  
**Date of Audit:** November 18, 2026  
**Auditor / Lead:** Corresponding Author & Principal Investigator  
**Workspace:** `e:\AIUB\Research\Breast Cancer Paper`

---

## 1. Executive Summary & Repository Archeology

An exhaustive inspection of the working environment reveals a critical structural duality:
1. **The Submitted Manuscript (ICCIT 2026 Submission ID 642):**
   - Title: *Diagnosing Test-Time Adaptation Failures in Multi-Instance Learning for Breast Cancer Histopathology* (with alternate working title: *When Does Test-Time Adaptation Fail for Multi-Instance Learning? A Diagnostic Study on Breast Cancer Histopathology*).
   - Core premise: Evaluates Test-Time Adaptation (TTA: TENT, TTT, CORAL, Hybrid) on Multi-Instance Learning (AB-MIL, TransMIL, CLAM-SB) transferred from BACH (source domain) to BreaKHis (target domain).
   - Core claim: Reports a "null result" where TTA fails to improve performance, attributing the failure to label-space mismatch (BACH 4-class vs. BreaKHis 2-class) and random head re-initialization.
2. **The In-Situ Codebase & Checkpoints (MAGNET Project):**
   - Title: *MAGNET: Multi-scale Attention-based Graph Network with Ensemble Transfer Learning for Breast Cancer Histopathology*.
   - Implemented in: `MAGNET_Final_fixed.ipynb`, `MAGNET.ipynb`, `MAGNET_notebook.ipynb`, `MAGNET_notebook-v2.ipynb`, and `MAGNET_Research.ipynb`.
   - Core premise: End-to-end multi-scale architecture processing BreaKHis 4-magnification biopsies (40X, 100X, 200X, 400X) with Macenko stain normalization, attention fusion, graph neural network convolutions, focal loss, Monte-Carlo dropout uncertainty quantification, Grad-CAM, and patient-wise 5-fold cross-validation.
   - All 20 saved model checkpoints in `checkpoints/`, all metric logs in `results/`, and all PNG plots in `plots/` belong directly to MAGNET.

This inventory documents all existing files, establishes their exact lineage, verifies their physical parameters, and pinpoints where artifacts are missing or decoupled.

---

## 2. Computational Environment & System Specifications

- **Operating System:** Windows 11 (build architecture: AMD64 / x86_64)
- **Primary Python Interpreter:** Python 3.x (Anaconda base environment `E:\Anaconda`)
- **PyTorch Installation:** PyTorch 2.14.0+cpu (CPU-only build currently active in default PATH)
- **GPU Hardware:** NVIDIA GeForce GTX 1650 (Driver: 621.24, CUDA Compute Capability: 7.5, Total VRAM: 4,096 MiB / 4 GB)
- **CUDA Acceleration:** CUDA UMD Version 13.5 present on host, currently unlinked to CPU PyTorch binary. For heavy retraining or deep ViT feature extraction, an accelerated CUDA PyTorch environment or dedicated compute cluster is required.
- **Key Python Dependencies Available:** `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `pypdf`, `timm`.

---

## 3. Directory Structure & File Manifest

```
E:\AIUB\Research\Breast Cancer Paper\
├── Conference Management Toolkit - View review.pdf  (Official ICCIT 2026 CMT Reviews)
├── MAGNET_Final_fixed.ipynb                         (Active primary notebook: MAGNET pipeline)
├── MAGNET_Final.ipynb                               (Precursor notebook to Final_fixed)
├── MAGNET.ipynb                                     (Full experimental notebook with TTA/Augmentation)
├── MAGNET_notebook-v2.ipynb                         (Intermediate revision)
├── MAGNET_notebook.ipynb                            (Initial full draft)
├── MAGNET_Research.ipynb                            (Exploratory research draft)
├── fix_notebook.py                                  (Script patching OneCycleLR & NaN checks)
├── folder_tree.txt                                  (Raw file system directory tree dump)
├── checkpoints/                                     (20 PyTorch model checkpoints, ~1.6 GB total)
├── data/                                            (BreaKHis raw dataset: 7,909 images)
├── logs/                                            (Execution logs directory - currently empty)
├── Manuscript/                                      (Submission LaTeX source, bibtex, and PDF figures)
│   ├── original_iccit2026/                          (Safely preserved original submission archive)
│   ├── main.tex                                     (ICCIT 2026 submitted paper)
│   ├── main-2.tex                                   (Alternative draft with expanded abstract)
│   ├── main old.tex                                 (Original working draft)
│   ├── references.bib                               (BibTeX bibliography file)
│   ├── references old.bib                           (Backup BibTeX file)
│   ├── IEEEtran.cls & IEEEtran.bst                  (IEEE conference template files)
│   └── *.pdf / *.png                                (7 figure assets)
├── paper/figures/                                   (Figure source copies and draw.io diagrams)
├── plots/                                           (13 PNG plots generated by MAGNET pipeline)
├── results/                                         (15 CSV, JSON, and LaTeX table summaries from MAGNET)
├── research_audit/                                  (Forensic analysis, audit, and upgrade blueprints)
└── experiments/                                     (New reproducible experiment workspace)
```

---

## 4. Dataset Inventory & Verification

### 4.1 BreaKHis Dataset (`data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast`)
Verified via automated physical disk traversal:
- **Total Images on Disk:** Exactly **7,909** PNG images (0 missing, 0 corrupt).
- **Patient Cohort:** Exactly **82 distinct patients** (24 Benign, 58 Malignant).
- **Magnification Distribution Breakdown:**
  | Magnification Factor | Benign Images | Malignant Images | Total Images | Class Balance (% Malignant) |
  | :--- | :---: | :---: | :---: | :---: |
  | **40X** | 625 | 1,370 | 1,995 | 68.67% |
  | **100X** | 644 | 1,437 | 2,081 | 69.05% |
  | **200X** | 623 | 1,390 | 2,013 | 69.05% |
  | **400X** | 588 | 1,232 | 1,820 | 67.69% |
  | **Overall Total** | **2,480** | **5,429** | **7,909** | **68.64%** |

- **Subtype Granularity:**
  - *Benign (4 subtypes):* Adenosis (A), Fibroadenoma (F), Phyllodes Tumor (PT), Tubular Adenoma (TA).
  - *Malignant (4 subtypes):* Ductal Carcinoma (DC), Lobular Carcinoma (LC), Mucinous Carcinoma (MC), Papillary Carcinoma (PC).

### 4.2 BACH (ICIAR 2018) Dataset
- **Status in Workspace:** **Physically absent.**
- Only referenced in `Manuscript/main.tex` and represented by one static figure (`tsne_resnet50_bach.pdf`).
- Ground-truth BACH characteristics from literature: 400 microscopy images (2048×1536 px), 4 balanced classes (Normal, Benign, InSitu, Invasive Carcinoma, 100 images each).
- Offline feature caches mentioned in paper (ResNet-50 102.5 GB, UNI 51.3 GB) are **not present** on the local drive.

---

## 5. Model Checkpoint Inventory (`checkpoints/`)

Total volume: ~1.6 GB across 20 `.pth` files:
1. **Core MAGNET Checkpoints:**
   - `best_model.pth` (144.4 MB): Optimal model from fixed-split training.
   - `fold_1.pth` through `fold_5.pth` (144.4 MB each): 5-fold cross-validation patient-wise checkpoints.
2. **Ablation Checkpoints:**
   - `ablation_wo_GNN.pth` (143.1 MB): MAGNET variant without Graph Neural Network layers.
   - `ablation_wo_Aux.pth` (144.4 MB): MAGNET variant without auxiliary subtype classification head.
   - `ablation_wo_Attn.pth` (144.4 MB): MAGNET variant replacing Multi-Head Attention with mean pooling.
   - `ablation_wo_Both.pth` (143.1 MB): Minimalist baseline without GNN or Attention.
3. **Single-Backbone & Fusion Baselines:**
   - `baseline_resnet34_40X.pth` (85.3 MB), `baseline_resnet34_200X.pth` (85.3 MB)
   - `baseline_efficientnet_b0_40X.pth` (16.3 MB)
   - `baseline_mobilenetv3_large_100_40X.pth` (17.0 MB)
   - `concat_baseline.pth` (131.2 MB): Simple feature concatenation across 4 magnifications.
   - `baseline_cv_fold_1.pth` through `baseline_cv_fold_5.pth` (16.3 MB each): 5-fold CV models for EfficientNet-B0@40X.

---

## 6. Experimental Results & Numerical Verification (`results/`)

All files in `results/` map directly to the MAGNET experiment:
- `all_results.json`: Comprehensive JSON ledger containing parameter counts (35.89M total, 31.01M backbone), fixed-split test metrics (Accuracy: 0.7692, AUROC: 0.8333, F1: 0.7692 on 13 test patients), and 5-fold CV aggregate metrics (Binary Acc: 0.6375 ± 0.1950, AUROC: 0.6925 ± 0.1100).
- `ablation_results.csv`: Documents degradation when GNN, Aux head, or Attention is removed.
- `baseline_comparison.csv` & `table_baselines.tex`: Fixed-split comparison against ResNet-34, EfficientNet-B0, MobileNetV3, and ConcatFusion.
- `kfold_results.csv` & `cv_confidence_intervals.json`: Per-fold metrics showing substantial fold-to-fold variance (Fold 1: 0.5294, Fold 2: 0.4706, Fold 3: 0.6875, Fold 4: 0.7500, Fold 5: 0.7500).
- `statistical_tests.json`: McNemar tests comparing MAGNET against baselines (revealing non-significant p-values due to the small 13-patient test sample).

---

## 7. Manuscript & Presentation Artifacts (`Manuscript/` and `paper/figures/`)

- `main.tex`: 432 lines. Defines 5 tables and 6 figures.
- `references.bib`: 20 entries. Contains severe citation bugs identified by Reviewer #3 (STAMP misattributed to arXiv:2309.07778, which is the UNI preprint).
- **Embedded Figures:**
  - `fig_method.png`: High-level diagram generated from `paper/figures/draw.drawio.txt`.
  - `dataset_eda.pdf`: Contains the critical factual error stating BreaKHis has 31,636 images.
  - `main_results.pdf`: Visualizes the S1/S2 transfer accuracies (0.4590 on BreaKHis).
  - `ablation_results.pdf`: Visualizes the A1-A6 ablation study including the erroneous 0.0933 accuracy.
  - `attention_maps.pdf`: Visualizes patch-level attention weights.
  - `backbone_comparison.pdf`: ResNet-50 vs. UNI on BACH.
  - `tsne_resnet50_bach.pdf`: 2D t-SNE projection of BACH features.

---

## 8. Critical Lineage Disconnect: The Two Tracks

| Attribute | Track A (Submitted Manuscript) | Track B (Repository Implementation) |
| :--- | :--- | :--- |
| **Project Identity** | TTA-MIL Failure Diagnostic Study | MAGNET Multi-Scale Network |
| **Primary Methodology** | TENT, TTT, CORAL, Hybrid TTA on MIL | Multi-mag Attention Fusion + Graph Conv |
| **Source Domain** | BACH (400 images, 4 classes) | None (Trained directly on BreaKHis) |
| **Target Domain** | BreaKHis 200X (falsely cited as 7,909) | BreaKHis (all 4 magnifications, 82 patients) |
| **Code Availability** | Disconnected / Not preserved in repo | Fully present (`MAGNET_Final_fixed.ipynb`) |
| **Checkpoints** | None preserved | 20 `.pth` files preserved |
| **Status** | Rejected at ICCIT 2026 | Functional experimental pipeline |

**Conclusion of Reconnaissance:**
The repository holds a complete, sophisticated multi-scale deep learning project (MAGNET) on BreaKHis, while the rejected manuscript (Track A) attempted to present a negative/diagnostic TTA study on BACH→BreaKHis with artificial constraints that guaranteed failure. The scientific upgrade mission must resolve this disconnect, eliminate all data and evaluation bugs, and produce a defensible, reproducible, high-impact manuscript.
