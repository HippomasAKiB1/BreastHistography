# Dataset Specification & Preprocessing: BreaKHis & BACH

## 1. Primary Dataset: BreaKHis v1

The Breast Cancer Histopathological Image Classification (BreaKHis) database consists of 7,909 microscopic biopsy images collected from **82 clinical patients** at the P&D Laboratory, Brazil. Tissue sections were stained with hematoxylin and eosin (H&E), prepared for surgical biopsy, and imaged under a clinical Olympus BX-50 microscope with a relay lens magnification of 3.3X at four optical magnification tiers.

### 1.1 Verified Image and Patient Counts

```
======================================================================
Official BreaKHis v1 Verified Distribution
======================================================================
Cohort: 82 Patients (24 Benign, 58 Malignant)
Total Histopathology Images: 7,909
Image Resolution: 700 x 460 pixels (24-bit RGB)
======================================================================
Magnification  | Benign Images | Malignant Images | Total Images
40X            | 625           | 1,370            | 1,995
100X           | 644           | 1,437            | 2,081
200X           | 623           | 1,390            | 2,013
400X           | 588           | 1,232            | 1,820
======================================================================
Total Images   | 2,480 (31.4%) | 5,429 (68.6%)    | 7,909
======================================================================
```

### 1.2 Histological Subtypes

The dataset contains 4 benign and 4 malignant histological subtypes:

| Subtype Group | Code | Subtype Name | Images | Patients |
| :--- | :--- | :--- | :--- | :--- |
| **Benign** | A | Adenosis | 444 | 4 |
| | F | Fibroadenoma | 1,014 | 10 |
| | PT | Phyllodes Tumor | 453 | 3 |
| | TA | Tubular Adenoma | 569 | 7 |
| **Malignant** | DC | Ductal Carcinoma | 3,451 | 38 |
| | LC | Lobular Carcinoma | 626 | 5 |
| | MC | Mucinous Carcinoma | 792 | 9 |
| | PC | Papillary Carcinoma | 560 | 6 |
| **Total** | | **8 Subtypes** | **7,909** | **82** |

---

## 2. Preventing Patient-Level Data Leakage

A critical flaw in early histopathology literature is **random image-level splitting**. Because a single patient biopsy contributes multiple images at various magnifications, randomly splitting images across training and testing partitions places images from the same physical slide in both sets. This causes severe test contamination, inflating reported accuracies by memorizing patient-specific staining and stromal morphology.

### 2.1 Our Leakage-Proof Patient Partitioning Strategy
1. **Patient-Level Stratification**: The 82 patients are partitioned into 70% Train (57 patients), 15% Validation (12 patients), and 15% Test (13 patients).
2. **Zero Overlap**: All images belonging to a patient exist strictly within one split:
   $$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset, \quad \mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset, \quad \mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset$$
3. **5-Fold Patient-Stratified Cross-Validation**: All validation folds are stratified by patient binary class label, ensuring balanced representation across folds without patient leakage.

---

## 3. Optical Density Stain Normalisation (Macenko Algorithm)

Color variability arising from differing staining protocols, reagent batches, and slice thicknesses introduces non-biological variance. We apply Macenko-style optical density normalisation:

```python
def macenko_norm(img_pil: Image.Image) -> Image.Image:
    img = np.array(img_pil, dtype=np.float32)
    img = np.clip(img, 1.0, 255.0)
    # Convert to Optical Density (OD) space: OD = -log10(I / 255)
    od = -np.log(img / 255.0)
    for c in range(3):
        m = od[..., c].mean()
        s = od[..., c].std() + 1e-6
        od[..., c] = (od[..., c] - m) / s * 0.3 + 0.4
    img_norm = np.exp(-od) * 255.0
    return Image.fromarray(np.clip(img_norm, 0, 255).astype(np.uint8))
```

---

## 4. Secondary Source Domain: BACH Dataset

For out-of-domain transfer and cross-magnification generalisation experiments:
- **Dataset**: Grand Challenge on BreAst Cancer Histology (BACH).
- **Images**: 400 high-resolution annotated microscopy images ($2048 \times 1536$).
- **Classes**: 4 balanced classes ($n = 100$ each): Normal, Benign, In-situ Carcinoma, and Invasive Carcinoma.
- **Role**: Source domain pretraining for domain adaptation and cross-dataset robustness benchmarks.
