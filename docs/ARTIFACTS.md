# Model Checkpoints & Artifacts Registry

This document records the official checksums, parameters, file sizes, and architecture specifications for all trained model checkpoints in the **BreastHistography** study.

> **Note on Large Files**: GitHub rejects file uploads exceeding 100 MB. Several full MAGNET checkpoints range from 125 MB to 138 MB due to multi-branch vision encoders. For this reason, `.pth` files are tracked via external releases and verified using the cryptographic SHA256 checksums below.

---

## 1. Verified Checkpoints Registry

| Checkpoint Name | Size (MB) | Parameters | Architecture / Role | SHA-256 Checksum |
| :--- | :---: | :---: | :--- | :--- |
| `best_model.pth` | 137.71 MB | 36.1M | **Primary MAGNET Model** (4-branch + Attention + GNN + Aux) | `BD5B26BCD85EEB9CD3A4B50BE4AB02368B26F4595518C198BA685F209B229278` |
| `concat_baseline.pth` | 125.10 MB | 32.8M | Multi-scale Concat-Fusion Baseline | `143316D22DE3D2FC5672A478C30D66D326410992BA3C346456BDF1BAFE002CB3` |
| `ablation_wo_GNN.pth` | 136.46 MB | 35.7M | Ablation: MAGNET without Batch GNN | `020598BAB7FA426AE2A6A5DB8B40B32DCABBC1806B9413767D8DB69CF2B180CC` |
| `ablation_wo_Aux.pth` | 137.68 MB | 36.1M | Ablation: MAGNET without Aux Supervision | `0B7E4EBB5DBC53D9AF2334DEA1B9CFE1EDF084609D58F4EF49A094C6ECE5CBF5` |
| `ablation_wo_Attn.pth` | 137.72 MB | 36.1M | Ablation: MAGNET without Multihead Attention | `EA66E62E82FA07E1BFE5EB3218F79D2F2701EF7A15054639AA540B4B4563C2D0` |
| `ablation_wo_Both.pth` | 136.43 MB | 35.7M | Ablation: MAGNET without Attention & GNN | `B6253DEACA19DB999AD55210007EC7559BC76DEDC917E3D6F65CEA12B89CF11C` |
| `fold_1.pth` | 137.71 MB | 36.1M | 5-Fold Cross-Validation: Fold 1 | `234C09595164E992C769D7F78B8EC7B86ADF8A87369990DB3EC7EEDD3DDF009D` |
| `fold_2.pth` | 137.71 MB | 36.1M | 5-Fold Cross-Validation: Fold 2 | `3DD7F3AFAE60DD83CCE5DF50B20E2EE5216078367BD71A488A765058E52CC8B3` |
| `fold_3.pth` | 137.71 MB | 36.1M | 5-Fold Cross-Validation: Fold 3 | `8652690D15B9B1B45C445C7BD056FEB7F5FB58AF368EAB7C2B8A5D59D0623B9E` |
| `fold_4.pth` | 137.71 MB | 36.1M | 5-Fold Cross-Validation: Fold 4 | `301F950BBF49ADBA9CC92693B80C69070F7FD443F39275638BB863B5BA736488` |
| `fold_5.pth` | 137.71 MB | 36.1M | 5-Fold Cross-Validation: Fold 5 | `AA5ABF12BD8FD004060BA745BE27A96AEC64E16D7555804F842E53E2230306A7` |
| `baseline_resnet34_200X.pth` | 81.34 MB | 21.3M | Unimodal ResNet-34 on 200X | `376A7DE5117EFEEE57475462479255B8EA3EFA4A4E3610EDA6376C6C69161EA7` |
| `baseline_resnet34_40X.pth` | 81.34 MB | 21.3M | Unimodal ResNet-34 on 40X | `E208445855B6F16DE34D2E99DDD4AFD75F3C6DCB342DD3F4563B0F14B5E3326A` |
| `baseline_mobilenetv3_large_100_40X.pth` | 16.24 MB | 4.2M | Unimodal MobileNetV3-Large on 40X | `EC380C613936407CB7F259FA9A1B2C3E1DDC8B5F6827675CCBA798EBBE7B055D` |
| `baseline_efficientnet_b0_40X.pth` | 15.59 MB | 4.0M | Unimodal EfficientNet-B0 on 40X | `75ADAB5AE4C1D8C564602D33843BCCCE1640C8A5805CFD9E740197EED4F7B939` |

---

## 2. Checkpoint Verification Script

To verify that local checkpoints match the official releases:

```bash
# On Linux/macOS:
sha256sum checkpoints/best_model.pth

# On Windows PowerShell:
Get-FileHash -Path checkpoints/best_model.pth -Algorithm SHA256
```

---

## 3. External Release Hosting (Hugging Face / Zenodo)

To download pre-trained weights from Hugging Face Hub:
```python
from huggingface_hub import hf_hub_download

checkpoint_file = hf_hub_download(
    repo_id="HippomasAKiB1/BreastHistography-weights",
    filename="best_model.pth",
    local_dir="checkpoints"
)
```
