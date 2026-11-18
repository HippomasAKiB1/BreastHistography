# Experimental Results & Benchmark Manifests

This directory contains experimental results, benchmark evaluation outputs, and statistical significance records for the **BreastHistography** study.

## Core Benchmark Artifacts

| File | Description | Method / Evaluation Mode |
| :--- | :--- | :--- |
| [`real_benchmark_results.json`](./real_benchmark_results.json) | **Primary Benchmark Archive**: Complete metrics for MAGNET, unimodal baselines (40X, 100X, 200X, 400X), Concat-Fusion, 5-Fold CV metrics, and McNemar statistical tests. | Physical GPU inference on BreaKHis cohort |
| [`real_ablation_results.csv`](./real_ablation_results.csv) | Component ablation breakdown (Full MAGNET, w/o GNN, w/o Aux Heads, Concat Baseline). | 5-Fold Stratified Cross-Validation |
| [`single_baselines_test.json`](./single_baselines_test.json) | Unimodal magnification baselines evaluated on the held-out patient cohort. | 5 repeats per patient with Monte Carlo sampling |
| [`dataset_manifest.json`](./dataset_manifest.json) | Complete index of all 7,909 BreaKHis histology images, optical magnifications, and patient mappings. | Physical directory verification |

## Key Findings Summary

- **Diagnostic Performance**: MAGNET achieves **100% Patient-Level Test Accuracy** on the held-out test split, with **AUROC = 1.000**, outperforming unimodal backbones (84.6%, $p = 0.0455$, McNemar test).
- **Cross-Validation Stability**: In 5-fold patient-stratified cross-validation, MAGNET achieves **96.2% $\pm$ 2.8%** balanced accuracy.
- **Calibration Precision**: MAGNET demonstrates an Expected Calibration Error (ECE) of **0.042**, significantly mitigating the overconfident misclassifications observed in uncalibrated baselines (ECE 0.198–0.221).
