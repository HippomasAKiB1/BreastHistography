# Reviewer Feedback Forensic Analysis

**Manuscript:** Paper ID 642 (*Diagnosing Test-Time Adaptation Failures in Multi-Instance Learning for Breast Cancer Histopathology*)  
**Venue:** ICCIT 2026 (Microsoft CMT Official Review)  
**Status:** Rejected  
**Audit Lead:** Corresponding Author & Principal Investigator  

---

## 1. Official Review Extraction & Itemized Transcript

### Reviewer #1
> **Comment:**  
> *"More latest works should be included."*

### Reviewer #3
> **Comment:**  
> 1. *"The null result is built into the design: replacing the head with a random 2-class layer guarantees that LayerNorm-only TTA cannot work. The obvious fix (BACH collapsed to binary), which the paper itself proposes, is never run."*  
> 2. *"BreakHis has 7,909 images in total and 2,013 at 200X; the stated 31,636 is exactly 4 × 7,909, and '200X subset (n ≈ 7,909)' is the full dataset."*  
> 3. *"A binary classifier that collapses to one class scores the class prevalence (about 0.31 or 0.69), so the 0.0933 'single-class collapse' accuracy in Table VI is impossible and points to an evaluation bug."*  
> 4. *"S1 is the fold used for checkpoint selection, not a held-out set, so its 1.0000 is not informative. A6 should state whether its 180 fine-tuning images are excluded from the 600 it is evaluated on."*  
> 5. *"UNI was pretrained on over 100 million tiles from over 100,000 slides, not '>100,000 tiles'. Reference [18] is cited as the TTA method STAMP, but its listed title describes a general pathology system."*

### Reviewer #4
> **Comment:**  
> *"While the root-cause diagnosis of label-space mismatch is valuable, the paper stops at identifying the failure without implementing or validating a proposed fix (e.g., joint head- and-normalization adaptation). Testing cross-domain adaptation across only one source-to-target dataset combination is insufficient to establish broader claims about TTA behavior across diverse computational pathology workflows and staining/scanner variations."*

---

## 2. Structured Taxonomical Breakdown of Review Criticisms

| Feedback Category | Primary Issue Identified | Reviewer(s) | Severity Rating |
| :--- | :--- | :---: | :---: |
| **Scientific Contribution & Novelty** | The paper presents a manufactured negative result rather than a true discovery; stops at diagnosing failure without demonstrating any working adaptation or positive contribution. | R3, R4 | **Critical** |
| **Methodological Integrity** | Discarding the classifier head and replacing it with random weights mathematically guarantees that LayerNorm-only adaptation fails. The experimental setup itself caused the failure. | R3 | **Critical** |
| **Dataset Validity & Statistics** | Hallucinated total dataset count (31,636 instead of 7,909) and claimed the 200X subset had 7,909 images (which is the size of the entire multi-magnification corpus). | R3 | **Critical** |
| **Evaluation Integrity & Metrics** | Reported a binary classification accuracy of 0.0933 under single-class collapse, which is mathematically impossible given class prevalences of 0.31 / 0.69, indicating a fatal evaluation bug. | R3 | **Critical** |
| **Data Leakage & Split Protocol** | S1 evaluated the exact fold used for checkpoint selection (yielding a tautological 1.0000); A6 failed to establish disjoint train/eval splits for fine-tuning. | R3 | **High** |
| **Experimental Scope & Generalization** | Only a single source-to-target pair (BACH → BreaKHis) was tested; insufficient to make broad generalizations regarding TTA in histopathology. | R4 | **High** |
| **Literature Rigor & Citations** | Foundation model pretraining scale misstated by orders of magnitude (>100k tiles vs. >100M tiles); citation [18] misattributed to STAMP while citing an unrelated paper. | R1, R3 | **Medium** |
| **Actionability & Validation** | Proposed fixes (collapsing BACH to binary; joint head-and-norm adaptation) were discussed in text but never empirically implemented or verified. | R3, R4 | **Critical** |

---

## 3. Deep Forensic Evaluation of Criticisms

### 3.1 Point 1: The Manufactured Null Result (Reviewers #3 & #4)
- **Scientific Reality:** In TTA methods such as TENT (Wang et al., ICLR 2021), only the affine parameters $\gamma, \beta$ of normalization layers (BatchNorm or LayerNorm) are optimized by minimizing Shannon entropy:
  $$\mathcal{L}_{\text{ent}} = -\sum_{c} p(c) \log p(c)$$
  The classification head $W \in \mathbb{R}^{C \times D}$ maps the feature space into $C$ class logits. When the 4-class head from BACH is discarded and a new 2-class head $W_{\text{rand}} \sim \mathcal{N}(0, \sigma^2)$ is attached with frozen feature encoders, the logits are random linear projections of the feature representations. Updating only the scalar gain and bias per LayerNorm channel cannot rotate the hyperplane to separate classes.
- **Root Cause:** The authors framed this structural inability as a "diagnostic finding," yet simultaneously wrote in Section V-D: *"A viable solution is collapsing BACH's 4 classes into 2... then transfer directly to BreakHis's benign/malignant taxonomy... allowing the trained 2-class head to be transferred as-is."* By mentioning the obvious fix but refusing to run it, the paper alienated the reviewers.
- **Verdict:** **100% VALID.** The critique is devastating and scientifically incontrovertible.

### 3.2 Point 2: BreaKHis Dataset Statistics Hallucination (Reviewer #3)
- **Scientific Reality:** Our disk verification confirmed:
  - 40X: 1,995 images
  - 100X: 2,081 images
  - 200X: 2,013 images
  - 400X: 1,820 images
  - Total across all magnifications: exactly **7,909 images**.
  The paper stated: *"The full dataset spans 31,636 images across all magnifications. Cross-domain evaluation uses the 200X subset (n ≈ 7,909)"*.
- **Root Cause:** The author multiplied 7,909 by 4 ($7{,}909 \times 4 = 31{,}636$), erroneously assuming that 7,909 was the count per magnification rather than the sum, and then claimed the 200X subset contained 7,909 images.
- **Verdict:** **100% VALID.** A catastrophic factual error that destroys reviewer confidence.

### 3.3 Point 3: Mathematical Impossibility of 0.0933 Accuracy (Reviewer #3)
- **Scientific Reality:** For any binary classification problem with ground-truth labels $y \in \{0, 1\}$ and class prior $P(y=1) = p$, a trivial or degenerate classifier predicting a constant class $\hat{y} = c \in \{0, 1\}$ for all samples achieves an accuracy of:
  $$\text{Acc}_{\text{const}} = \begin{cases} 1 - p, & \text{if } c = 0 \\ p, & \text{if } c = 1 \end{cases}$$
  In BreaKHis 200X, $p = \frac{1390}{2013} \approx 0.6905$, and $1 - p \approx 0.3095$. In the 600-sample stratified subset, the class counts are exactly 186 Benign (31.0%) and 414 Malignant (69.0%). Thus, complete collapse to Benign yields 0.3100 accuracy; complete collapse to Malignant yields 0.6900 accuracy.
  An accuracy of **0.0933** ($56 / 600$) is mathematically impossible for a single-class collapse on this dataset.
- **Root Cause:** A fatal bug in the evaluation script. Likely causes:
  1. Evaluating 4-class logits against binary ground truth without proper indexing.
  2. Inverted label index mapping combined with an unnormalized denominator.
  3. Mismatched sample indices between predictions and ground-truth arrays.
- **Verdict:** **100% VALID.** Proves the evaluation code was bugged.

### 3.4 Point 4: Circular Evaluation on S1 & Lack of Disjoint Splitting in A6 (Reviewer #3)
- **Scientific Reality:**
  - In Table 5 (Scenario S1), the paper reports 1.0000 across all models and methods on the BACH validation set ($n=80$). Section III-D explicitly states: *"the fold with the highest validation accuracy provides the model for all cross-domain experiments."* Testing a model on the validation fold that determined its selection is textbook circular reporting.
  - In Table 6 (Condition A6), 180 images (30% of the 600-sample subset) were fine-tuned. The paper never clarifies whether the reported accuracy (0.8533 / 0.9250) was computed on the remaining 420 held-out images or on the full 600 images (inclusive of the 180 training samples).
- **Verdict:** **100% VALID.** Represents methodological and reporting carelessness.

### 3.5 Point 5: Literature & Citation Inaccuracies (Reviewer #1 & Reviewer #3)
- **Scientific Reality:**
  - In Section III-B, the paper states: *"UNI: ViT-L/16 pretrained on >100,000 H&E pathology tiles"*. In reality, Chen et al. (*Nature Medicine* 2024) trained UNI on over **100 million tiles** extracted from **100,426 whole-slide images**. Stating ">100,000 tiles" understated the pretraining scale by a factor of 1,000.
  - In `references.bib`, entry `[18]` is cited as STAMP (a test-time adaptation technique for slide encoders), but its title is given as *"A general-purpose AI system for pathology at scale"*, which is actually the arXiv title for UNI / Chief!
  - Reviewer #1 rightly requested modern literature (e.g., 2024–2026 foundation models and adaptation frameworks).
- **Verdict:** **100% VALID.** Direct evidence of unverified citations and sloppy literature review.

### 3.6 Point 6: Scope Limitations & Absence of Validated Fixes (Reviewer #4)
- **Scientific Reality:** Testing a single cross-domain transfer (BACH → BreaKHis) under an artificially broken setup, declaring that TTA fails, and offering no positive solution or multi-dataset validation provides minimal utility to the scientific community.
- **Verdict:** **100% VALID.** Confirms that competitive venues require constructive solutions and rigorous multi-condition validation.

---

## 4. Master Response Strategy & Remediation Matrix

| Reviewer Item | Action Required | Technical Solution in Upgrade Mission |
| :--- | :--- | :--- |
| **R3.1: Null result built into design** | Implement the obvious fix & head adaptation | (1) Establish direct label alignment: Map BACH 4-class to binary (Normal+Benign vs. InSitu+Invasive) so the trained 2-class head transfers directly. (2) Implement head-aware test-time adaptation (joint normalization + classifier alignment via prototype clustering / entropy sharpening). |
| **R3.2: BreaKHis image count error** | Correct all dataset numbers | Audit and rewrite all dataset tables to reflect the true ground truth: BreaKHis has 7,909 total images across 4 magnifications, exactly 2,013 images at 200X, from 82 patients (24 benign, 58 malignant). |
| **R3.3: Impossible 0.0933 accuracy bug** | Fix evaluation script and recompute metrics | Rebuild the evaluation metrics pipeline from scratch with automated unit tests: enforce binary confusion matrix invariants ($TP + TN + FP + FN = N$), verify that degenerate predictors strictly equal class prevalence ($0.31$ / $0.69$). |
| **R3.4: S1 circularity & A6 leakage** | Redesign splits to be strictly disjoint | (1) Report genuine out-of-fold cross-validation or a strictly held-out test cohort for source evaluation. (2) For fine-tuning baselines, enforce a strict train/test split (train on 180, evaluate strictly on the remaining 420 disjoint samples). |
| **R3.5: UNI tile count & STAMP citation** | Correct citations and literature figures | Correct UNI's pretraining volume to >100M patches from >100k WSIs. Replace corrupted citation [18] with the accurate STAMP citation or modern TTA pathology benchmarks. |
| **R4: Lack of working fix & single pair** | Provide positive adaptation + multi-mag evaluation | Move beyond negative diagnostics: demonstrate when and how TTA succeeds (aligned head, prototype alignment, multi-scale feature adaptation). Validate across multi-magnification shifts (40X, 100X, 200X, 400X) leveraging the rich BreaKHis dataset. |
| **R1: Modern literature** | Expand related work | Integrate 2024–2026 literature: modern pathology foundation models (UNI, CONCH, Virchow2, GigaPath) and state-of-the-art adaptation techniques (SAR, RoTTA, DePT, EcoTTA). |
