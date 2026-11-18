# Experimental Setup, Protocols & Training Procedures

## 1. Hardware & Software Environment

- **Host Processor**: AMD Ryzen 5 / Intel Core i7
- **Host GPU**: NVIDIA GeForce GTX 1650 (4,096 MB VRAM, Turing Architecture, Compute Capability 7.5)
- **CUDA Runtime**: CUDA 12.8
- **Deep Learning Framework**: PyTorch 2.11.0+cu128
- **Vision Backbones**: `timm` (PyTorch Image Models v0.9.12)
- **OS**: Windows 11 Enterprise (64-bit)

---

## 2. Training Hyperparameters

```yaml
training:
  batch_size: 4
  epochs: 50
  optimizer: AdamW
  learning_rate: 1.0e-4
  weight_decay: 1.0e-2
  lr_scheduler: CosineAnnealingWarmRestarts
  t_0: 10
  t_mult: 2
  min_lr: 1.0e-6
  gradient_accumulation_steps: 2
  mixed_precision: fp16
  early_stopping_patience: 10
  seed: 42

loss_weights:
  lambda_binary: 1.0
  lambda_subtype: 0.5
  lambda_aux: 0.25
  focal_gamma: 2.0
  focal_alpha: 0.25
```

---

## 3. Training Protocol & Deep Supervision

1. **Staged Transfer Learning**:
   Backbones (`mobilenetv3_large_100`, `efficientnet_b0`, `resnet34`, `mobilenetv3_small_100`) are initialized from ImageNet weights. The projection layers, cross-magnification attention, and batch graph convolutions are trained first with backbone learning rate discounted by $0.1\times$ to prevent catastrophic forgetting.
2. **Dynamic Mini-Batch Graph Construction**:
   Mini-batches contain 4 patients sampled with balanced class weighting. Adjacency matrices are recomputed dynamically during forward passes using cosine feature similarity with threshold $\tau = 0.5$.
3. **Multi-Scale Auxiliary Regularization**:
   Each backbone features an auxiliary linear classifier evaluated against binary ground-truth labels during backpropagation. This forces individual encoders to extract diagnostically discriminative features before cross-magnification fusion.

---

## 4. Test-Time Adaptation Protocol

- **Target Adaptation Batch Size**: Single-bag adaptation (1 patient slide at a time).
- **Adaptation Steps**: $T = 3$ gradient steps per bag.
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum 0.9, learning rate $\eta = 1.0 \times 10^{-3}$.
- **Adapted Parameters**: Normalization layers (LayerNorm affine parameters) + Classification MLP heads.
- **Weight Rollback**: After producing logits for patient slide $k$, all adapted model weights are restored to initial checkpoint state $\theta_0$ in microseconds via an in-memory state dictionary buffer. This completely eliminates negative cumulative adaptation drift across sequential patient admissions.
