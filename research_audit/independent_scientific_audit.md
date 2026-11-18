# Independent Scientific Audit & Forensic Review

**Lead:** Corresponding Author, Principal Researcher & Senior Scientific Auditor  
**Scope:** Comprehensive Audit of Manuscript ID 642 (*Diagnosing Test-Time Adaptation Failures...*), the Underlying Codebases (`MAGNET*.ipynb`), Experimental Artifacts, and Data Pipelines.

---

## 1. Executive Scientific Verdict

The rejected manuscript suffered from a fundamental flaw: **it constructed an artificial failure mode through an unsound experimental protocol and framed that failure as an inherent scientific diagnosis of Test-Time Adaptation (TTA) in Multi-Instance Learning (MIL).**

Beyond the fatal issues caught by Reviewers #3 and #4, our independent scientific audit has uncovered several deeper methodological, statistical, and empirical vulnerabilities that would guarantee immediate rejection at any competitive venue (e.g., IEEE TMI, JBHI, BIBM, MICCAI, CVPR/ICCV/ECCV workshops).

Simultaneously, our audit of the in-situ **MAGNET** codebase reveals an intact, methodologically advanced multi-scale graph attention system on BreaKHis that was never presented in the ICCIT manuscript. However, the MAGNET implementation has its own latent experimental vulnerabilities (e.g., small test-cohort sample size, high cross-validation variance) that must be audited and hardened.

---

## 2. In-Depth Audit of Track A: The ICCIT 2026 TTA Manuscript

### 2.1 The "Random Head" Methodology: Scientific Validity Analysis
- **What Was Done:**
  The source model was trained on BACH (4 classes: Normal, Benign, InSitu, Invasive). For cross-domain transfer to BreaKHis (2 classes: Benign, Malignant), the 4-class linear classification layer was stripped, a new 2-class `nn.Linear` layer was randomly initialized, and all feature extraction weights were frozen. TTA was then executed by updating solely the LayerNorm affine parameters ($\gamma, \beta$).
- **Independent Critique:**
  1. *Mathematical Futility:* LayerNorm operates channel-wise:
     $$\hat{z}_i = \frac{z_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i$$
     This scales and shifts individual feature dimensions. It represents an axis-aligned transformation. A random classification layer $W_{\text{rand}} \in \mathbb{R}^{2 \times D}$ defines an arbitrary hyperplane through feature space. No axis-aligned channel rescaling can rotate or reconfigure high-dimensional representations to align with arbitrary random hyperplane normal vectors.
  2. *Theoretical Contradiction:* The foundational premise of Test-Time Adaptation (Wang et al., 2021; Sun et al., 2020) is to correct for covariate shift $P(X)$ while assuming the semantic mapping $P(Y|X)$ remains approximately invariant. By randomizing the head, the authors destroyed $P(Y|X)$ entirely. Calling this a "failure of TTA" is scientifically invalid; it is a failure of random linear projection.
  3. *Unperformed Obvious Solution:* In breast pathology, the mapping between BACH and BreaKHis is clinically and ontologically canonical:
     $$\text{BACH}\{\text{Normal}, \text{Benign}\} \longrightarrow \text{Benign}$$
     $$\text{BACH}\{\text{InSitu}, \text{Invasive}\} \longrightarrow \text{Malignant}$$
     By training the source model on this clinically sound binary formulation, the head is preserved, $P(Y|X)$ is maintained, and genuine cross-domain distribution shifts can be tested. Leaving this unexecuted while acknowledging it in Section V-D invalidated the entire contribution.

### 2.2 Dataset Integrity & Dimensional Discrepancies
- **Magnification Mismatch:**
  - BACH consists of 400 high-resolution microscopy images acquired at a single optical magnification ($200\times$, $0.42\,\mu\text{m/px}$).
  - BreaKHis consists of 7,909 images acquired across four distinct magnifications ($40\times, 100\times, 200\times, 400\times$).
  - When the paper transferred from BACH to BreaKHis 200X, it evaluated patch bags extracted with identical stride (256×256 px, stride 128 px, max 32 patches). However, the optical resolution and visual field of view (FOV) between BACH and BreaKHis at 200X differ substantially due to different microscope optics and camera sensors (Motic vs. Leica). The paper failed to normalize for physical pixel dimensions ($\mu\text{m/px}$) or optical field of view.

### 2.3 The 0.0933 Evaluation Bug: Forensic Diagnosis
- In Table VI (Ablation study on a 600-sample stratified BreaKHis subset), the authors reported:
  - ResNet-50 No TTA: `0.0933` (9.33%)
  - ResNet-50 TENT: `0.0933`
  - ResNet-50 TTT: `0.0933`
  - ResNet-50 CORAL: `0.0933`
- *Mathematical Audit:*
  - On 600 samples, an accuracy of 0.0933 corresponds to exactly:
    $$0.093333... \times 600 = 56 \text{ correct predictions}$$
  - In BreaKHis 200X, the class distribution is ~31% Benign and ~69% Malignant. In 600 stratified samples, there are 186 Benign and 414 Malignant samples.
  - If a model predicts all Benign, accuracy is $\frac{186}{600} = 31.0\%$.
  - If a model predicts all Malignant, accuracy is $\frac{414}{600} = 69.0\%$.
  - How did 56/600 occur?
    1. *Hypothesis 1 (Inverted Class Mapping on Benign Subset):* If the script evaluated only on benign samples or evaluated an inverted prediction on a fraction of the cohort ($56 / 186 \approx 30.1\%$).
    2. *Hypothesis 2 (4-Class Argmax against 2-Class Ground Truth):* If the model output 4-class logits $[B, 4]$ and compared `preds.argmax(dim=-1)` (values $0, 1, 2, 3$) directly against binary labels ($0, 1$). Predictions of classes 2 and 3 would be scored as wrong automatically, drastically suppressing accuracy below the binary chance floor.
  - Either hypothesis proves that the evaluation pipeline was never verified against basic sanity checks.

### 2.4 Data Leakage and Circular Evaluation
- **Scenario S1 (Tautological Evaluation):**
  - Table V reports Accuracy = 1.0000, F1 = 1.0000, AUROC = 1.0000 for both ResNet-50 and UNI on the held-out validation set ($n=80$).
  - Section III-D explicitly documents that the model checkpoint used was selected based on the highest validation accuracy on this exact set. Selecting a checkpoint by $\arg\max_{\theta} \text{Acc}_{\text{val}}(\theta)$ and then reporting that exact validation accuracy as test performance is a fatal violation of experimental hygiene.
- **Condition A6 (Fine-Tuning Leakage):**
  - Condition A6 fine-tuned on 180 images (30% of the 600-sample subset) and reported 0.8533 (ResNet-50) and 0.9250 (UNI).
  - The manuscript does not specify whether evaluation was conducted on the remaining 420 disjoint samples or on the full 600 images. If evaluated on all 600, 30% of the test set was seen during training.

### 2.5 Literature and Citation Fabrications
- **STAMP Misattribution:** Reference [18] is cited as *"Chen et al., 2023. STAMP: A general-purpose AI system for pathology at scale, arXiv:2309.07778"*.
  - arXiv:2309.07778 is actually *"Towards a general-purpose foundation model for computational pathology"* (the UNI preprint).
  - STAMP is actually a completely different paper by Kather et al. / Lang et al. on biomarker prediction.
  - Conflating two unrelated landmark papers indicates sloppy citation curation.
- **UNI Pretraining Scale:** Describing UNI as trained on *">100,000 tiles"* when it was trained on **>100,000 whole-slide images yielding >100 million tiles** is a 1,000-fold inaccuracy.

---

## 3. In-Depth Audit of Track B: The In-Situ MAGNET Project

Our audit of `MAGNET_Final_fixed.ipynb`, `results/`, and `checkpoints/` reveals a substantial, well-engineered multi-scale architecture. However, several critical vulnerabilities must be addressed:

### 3.1 Patient-Wise vs. Image-Wise Splitting
- **The Good:** `MAGNET_Final_fixed.ipynb` explicitly groups images by patient ID (`SOB_B_...` and `SOB_M_...`), preventing image-level leakage between train and test splits. This is gold-standard practice in histopathology.
- **The Limitation:** The fixed-split test set contains only **13 patients** (6 Benign, 7 Malignant):
  - In a 13-patient test set, a single misclassified patient shifts accuracy by $\frac{1}{13} \approx 7.69\%$.
  - An accuracy of 0.7692 corresponds to exactly 10 correct out of 13 patients ($10/13 \approx 0.7692$).
  - Statistical power with $n=13$ is virtually zero. In `results/statistical_tests.json`, the McNemar test against ResNet-34 yields $p = 0.453$, and against MobileNetV3 yields $p = 0.125$. The manuscript cannot claim statistically significant superiority over baselines on this split.

### 3.2 5-Fold Cross-Validation Instability
- Looking at `results/kfold_results.csv`:
  - Fold 1: Test Acc = 0.5294 (9/17 correct)
  - Fold 2: Test Acc = 0.4706 (8/17 correct)
  - Fold 3: Test Acc = 0.6875 (11/16 correct)
  - Fold 4: Test Acc = 0.7500 (12/16 correct)
  - Fold 5: Test Acc = 0.7500 (12/16 correct)
  - **Mean Accuracy:** $0.6375 \pm 0.1950$ (95% CI).
- **Audit Finding:** The 95% confidence interval spanning $[0.4425, 0.8325]$ indicates extreme inter-fold variance. Fold 2 performs worse than random guessing on binary classification (0.4706). An honest reviewer will immediately attack this variance.
- **Root Cause:**
  1. Small cohort: 82 total patients divided into 5 folds means ~16 patients per fold.
  2. Severe class imbalance: 24 Benign vs. 58 Malignant. In each fold, there are only ~4–5 Benign patients. Misclassifying just 2 Benign patients drops fold accuracy drastically.
  3. Feature representation capacity: MobileNetV3/EfficientNet-B0 backbones trained with small batch sizes without foundation model pre-extraction exhibit high gradient noise.

---

## 4. Synthesis & Strategic Options for Upgrade

We face a critical strategic fork. To transform this rejected submission into a competitive, publication-ready paper, we must choose between two defensible paths:

### Option A: The Complete TTA-MIL Diagnostic & Recovery Paper
- **Title Concept:** *Closing the Test-Time Adaptation Gap in Multi-Instance Learning for Breast Cancer Histopathology: Preconditions, Label-Space Alignment, and Head-Aware Adaptation*
- **Core Narrative:**
  1. Acknowledge and rigorously formalize the failure modes of standard TTA (TENT, CORAL) when applied naively to MIL across institutional datasets.
  2. Formally demonstrate why random head re-initialization guarantees failure.
  3. **Implement and evaluate the complete proposed fix:**
     - Clinically aligned binary mapping: Source BACH mapped to Benign vs. Malignant, preserving the trained classifier head.
     - Head-Aware TTA (HAT-MIL): Joint adaptation of normalization statistics and classifier parameters via prototype clustering or contrastive pseudo-labeling.
     - Multi-magnification adaptation across BreaKHis domains.
  4. Fix all data statistics (7,909 total, 2,013 at 200X), correct the 0.0933 bug, fix all citations.

### Option B: The Unified Multi-Scale Attention & Adaptation Paper (MAGNET 2.0)
- **Title Concept:** *MAGNET: Multi-Scale Attention Graph Network with Domain-Adaptive Calibration for Breast Cancer Histopathology*
- **Core Narrative:**
  1. Focus on the core strength of this repository: multi-magnification histopathology representation learning.
  2. Integrate the multi-scale attention graph network with robust domain adaptation and calibration.
  3. Harden the 5-fold cross-validation results, expand patient-wise evaluation, and provide full interpretability (attention heatmaps, Grad-CAM, MC-dropout uncertainty).

### Option C: The Defensible Hybrid (Recommended)
- **Title Concept:** *Beyond Naive Test-Time Adaptation: Label-Aligned and Multi-Scale Feature Calibration for Cross-Domain Breast Cancer Histopathology*
- Unify the diagnostic rigor demanded by Reviewers #3 & #4 with the rich multi-magnification modeling and interpretability tools already built into the codebase.

---

## 5. Audit Checklist & Mandatory Pre-Submission Verifications

- [x] Disk verification of BreaKHis image and patient counts completed.
- [x] Formal mathematical proof of the 0.0933 impossibility established.
- [x] Reviewer criticisms categorized, validated, and mapped to solutions.
- [ ] Implement label-aligned cross-domain evaluation pipeline.
- [ ] Implement head-aware test-time adaptation baseline.
- [ ] Eliminate all circular validation reporting.
- [ ] Verify every numerical entry against reproducible code execution.
- [ ] Rebuild all manuscript figures and tables with verified data.
- [ ] Verify every reference, title, and pretraining statistic in BibTeX.
