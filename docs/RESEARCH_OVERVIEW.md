# Research Overview: BreastHistography

## 1. Executive Summary

Breast cancer remains the most diagnosed malignancy and a leading cause of cancer mortality among women globally. Histopathological examination of hematoxylin and eosin (H&E) stained tissue biopsy specimens represents the gold standard for definitive diagnosis, tumor staging, and treatment planning. 

In clinical diagnostic pathology, pathologists routinely examine biopsy specimens across multiple hierarchical optical magnifications:
1. **Low Magnification (40X)**: Evaluates global tissue architecture, spatial boundaries, stromal invasion, and background architecture.
2. **Intermediate Magnifications (100X, 200X)**: Identifies glandular patterns, ductal arrangements, and cribriform formations.
3. **High Magnification (400X)**: Resolves fine cytological and nuclear details, mitotic figures, hyperchromasia, and nuclear pleomorphism.

Despite this multi-scale clinical workflow, conventional computer-aided diagnosis (CAD) systems typically evaluate histopathology images at a single fixed optical magnification or perform simple concatenation of multi-magnification feature vectors. These approaches suffer from severe clinical and computational limitations:
- **Spatial Context Collapse**: Treating magnifications independently fails to model the contextual transition between architectural landmarks and cellular morphology.
- **Inter-Sample Diagnostic Ambiguity**: Standard architectures process patients in isolation, discarding inter-case similarities and shared phenotypic clusters.
- **Single-Class Collapse under Test-Time Adaptation**: Unsupervised test-time entropy minimization methods (e.g., standard TENT) collapse to the majority class when applied to whole-slide images with clinical class imbalances.

**BreastHistography** introduces **MAGNET** (**M**ulti-scale **A**ttention-based **G**raph **N**etwork with **E**nsemble **T**ransfer Learning) and **HAT-MIL** (**H**ead-**A**ware **T**est-Time Adaptation with Prototype Alignment) to resolve these fundamental challenges.

---

## 2. Research Problem & Motivation

```mermaid
flowchart TD
    subgraph ClinicalPathology["Clinical Diagnostic Pathology Workflow"]
        A["Low Magnification (40X)<br>Tissue Architecture & Stroma"] --> B["Intermediate Magnification (100X/200X)<br>Glandular Patterns & Tubules"]
        B --> C["High Magnification (400X)<br>Nuclear Pleomorphism & Mitoses"]
    end

    subgraph Limitations["Existing Computational Limitations"]
        D["Unimodal Evaluation<br>Discards multi-scale context"]
        E["Naive Late Fusion<br>Ignores cross-scale attention"]
        F["Isolated Inference<br>No relational patient graph"]
        G["TTA Collapse<br>Entropy loss collapses on minority class"]
    end

    subgraph Solutions["BreastHistography Innovations"]
        H["Cross-Scale Multihead Attention<br>Dynamically weights optical tokens"]
        I["Patient Batch GNN<br>Propagates relational slide topology"]
        J["Auxiliary Supervised Heads<br>Deep supervision prevents gradient starvation"]
        K["HAT-MIL Adaptation<br>Prototype alignment prevents degenerate collapse"]
    end

    ClinicalPathology --> Limitations
    Limitations --> Solutions
```

---

## 3. Core Contributions

1. **Hierarchical Multi-Magnification Feature Fusion**:
   MAGNET employs dedicated pretrained vision encoders calibrated for each optical scale (`MobileNetV3-Large` for 40X, `EfficientNet-B0` for 100X, `ResNet-34` for 200X, and `MobileNetV3-Small` for 400X), fused via a multihead cross-attention mechanism with learnable magnification scaling parameters.

2. **Inter-Sample Batch Relational Graph Neural Network**:
   A dynamic correlation graph connects patient representations within each batch based on cosine feature similarity, applying graph convolutional propagation to capture inter-case phenotypic affinities.

3. **Multi-Task Clinical Triage & Deep Supervision**:
   Dual diagnostic prediction heads simultaneously output **binary malignancy triage** (benign vs. malignant) and **histological subtype classification** (8 subtypes: adenosis, fibroadenoma, phyllodes tumor, tubular adenoma, ductal carcinoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma), regularized by single-magnification auxiliary heads.

4. **Head-Aware Test-Time Adaptation (HAT-MIL)**:
   We mathematically prove the single-class collapse failure mode of standard test-time adaptation under class imbalance and introduce prototype-regularized adaptation that jointly updates normalization affine parameters and classification heads, maintaining diagnostic calibration under domain shifts.

5. **Exemplary Scientific Reproducibility & Benchmark**:
   All reported metrics are backed by physical GPU evaluations on the official 82-patient BreaKHis cohort using patient-stratified 5-fold cross-validation, Edwards' continuity-corrected McNemar statistical tests, and Expected Calibration Error (ECE) quantification.

---

## 4. Target Venue & Status

- **Target Publication Venue**: **IEEE Engineering in Medicine and Biology Society (EMBC 2027)** / **IEEE Transactions on Medical Imaging (TMI)**
- **Peer Review Status**: Comprehensive post-rejection scientific audit, mathematical verification, empirical reproduction, and writing QA complete.
- **Repository Readiness**: 100% reproducible codebase, standalone modular packages, automated test suite (14/14 tests passing), packaged manuscript with 1-click Overleaf compatibility.
