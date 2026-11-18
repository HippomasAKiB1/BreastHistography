"""
generate_publication_figures.py: Generates verified, publication-quality vector figures
directly from experiments/results/real_benchmark_results.json.
Saves to both experiments/figures/ and Manuscript/.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Styling configuration for IEEE publications
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# Load master benchmark results
with open('experiments/results/real_benchmark_results.json', 'r') as f:
    bench = json.load(f)

def plot_dataset_eda():
    """Generates the verified dataset overview figure, correcting the 31,636 error."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    
    # Panel 1: BACH Class Distribution (Source)
    classes_bach = ['Normal', 'Benign', 'InSitu', 'Invasive']
    counts_bach = [100, 100, 100, 100]
    colors_bach = ['#2b5c8f', '#4682b4', '#d95f02', '#b22222']
    bars1 = axes[0].bar(classes_bach, counts_bach, color=colors_bach, edgecolor='black', linewidth=0.8, width=0.6)
    axes[0].set_title("(a) BACH Source Domain (n=400)", fontweight='bold')
    axes[0].set_ylabel("Number of Images")
    axes[0].set_ylim(0, 130)
    for b in bars1:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 3, f"{int(b.get_height())}", ha='center', fontsize=9)

    # Panel 2: BreaKHis Optical Magnification Distribution (Target)
    mags = ['40X', '100X', '200X', '400X']
    benign_counts = [625, 644, 623, 588]
    malignant_counts = [1370, 1437, 1390, 1232]
    total_mags = [b + m for b, m in zip(benign_counts, malignant_counts)]
    
    x = np.arange(len(mags))
    w = 0.35
    axes[1].bar(x - w/2, benign_counts, w, label='Benign (31.4%)', color='#4575b4', edgecolor='black', linewidth=0.8)
    axes[1].bar(x + w/2, malignant_counts, w, label='Malignant (68.6%)', color='#d73027', edgecolor='black', linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(mags)
    axes[1].set_title("(b) BreaKHis Target Magnifications (n=7,909)", fontweight='bold')
    axes[1].set_ylabel("Number of Images")
    axes[1].set_ylim(0, 1750)
    axes[1].legend(frameon=True, loc='upper left')
    for i, tot in enumerate(total_mags):
        axes[1].text(x[i], max(malignant_counts[i], benign_counts[i]) + 50, f"Tot: {tot}", ha='center', fontsize=8, fontweight='bold')

    # Panel 3: Patient Cohort Breakdown (82 Unique Patients)
    patient_classes = ['Benign Patients', 'Malignant Patients']
    patient_counts = [24, 58]
    colors_pat = ['#4575b4', '#d73027']
    bars3 = axes[2].bar(patient_classes, patient_counts, color=colors_pat, edgecolor='black', linewidth=0.8, width=0.5)
    axes[2].set_title("(c) Patient Cohort Distribution (N=82)", fontweight='bold')
    axes[2].set_ylabel("Number of Unique Patients")
    axes[2].set_ylim(0, 75)
    for b in bars3:
        axes[2].text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{int(b.get_height())} ({b.get_height()/82*100:.1f}%)", ha='center', fontsize=9)

    plt.tight_layout()
    fig.savefig("experiments/figures/dataset_eda.pdf")
    fig.savefig("Manuscript/dataset_eda.pdf")
    plt.close()
    print("✓ Saved verified dataset_eda.pdf")

def plot_ablation_results():
    """Generates the verified ablation chart directly from physical GPU evaluations."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    
    ablations = bench['ablations']
    names = [row['Configuration'].replace('Supervision', 'Sup.').replace('Cross-Mag ', '') for row in ablations]
    accuracies = [row['Accuracy'] for row in ablations]
    aurocs = [row['AUROC'] for row in ablations]
    f1s = [row['Macro-F1'] for row in ablations]
    
    x = np.arange(len(names))
    w = 0.26
    
    bars_acc = ax.bar(x - w, accuracies, w, label='Accuracy', color='#3182bd', edgecolor='black', linewidth=0.8)
    bars_auc = ax.bar(x, aurocs, w, label='AUROC', color='#e6550d', edgecolor='black', linewidth=0.8)
    bars_f1  = ax.bar(x + w, f1s, w, label='Macro-F1', color='#31a354', edgecolor='black', linewidth=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("Score")
    ax.set_ylim(0.2, 1.08)
    ax.set_title("Physical Ablation Study on Real Test Slides (N=13 Patients)", fontweight='bold')
    ax.legend(frameon=True, loc='upper left', ncol=3)
    
    for bar in bars_acc:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f"{yval:.3f}", ha='center', va='bottom', fontsize=7.5, rotation=90)
    for bar in bars_auc:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f"{yval:.3f}", ha='center', va='bottom', fontsize=7.5, rotation=90)

    plt.tight_layout()
    fig.savefig("experiments/figures/ablation_results.pdf")
    fig.savefig("Manuscript/ablation_results.pdf")
    plt.close()
    print("✓ Saved verified ablation_results.pdf")

def plot_main_results():
    """Generates the main benchmark & TTA comparison figure from real GPU metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    
    # Panel 1: Test-Set Benchmark (Baselines vs Concat vs MAGNET)
    models = ['ResNet34\n@40X', 'ResNet34\n@200X', 'EffNet-B0\n@40X', 'MobileNet\n@40X', 'Concat\nFusion', 'MAGNET\n(Ours)']
    accs = [
        bench['single_baselines_test']['ResNet-34 @ 40X']['accuracy'],
        bench['single_baselines_test']['ResNet-34 @ 200X']['accuracy'],
        bench['single_baselines_test']['EfficientNet-B0 @ 40X']['accuracy'],
        bench['single_baselines_test']['MobileNetV3-Large @ 40X']['accuracy'],
        bench['concat_baseline_test']['accuracy'],
        bench['fixed_split_test']['accuracy']
    ]
    aurocs = [
        bench['single_baselines_test']['ResNet-34 @ 40X']['auroc'],
        bench['single_baselines_test']['ResNet-34 @ 200X']['auroc'],
        bench['single_baselines_test']['EfficientNet-B0 @ 40X']['auroc'],
        bench['single_baselines_test']['MobileNetV3-Large @ 40X']['auroc'],
        bench['concat_baseline_test']['auroc'],
        bench['fixed_split_test']['auroc']
    ]
    
    x1 = np.arange(len(models))
    w1 = 0.35
    ax1.bar(x1 - w1/2, accs, w1, label='Accuracy', color='#4575b4', edgecolor='black', linewidth=0.8)
    ax1.bar(x1 + w1/2, aurocs, w1, label='AUROC', color='#d73027', edgecolor='black', linewidth=0.8)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(models, fontsize=8)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0.3, 1.05)
    ax1.set_title("(a) Architectural Benchmark on Test Patients", fontweight='bold')
    ax1.legend(frameon=True, loc='upper left')

    # Panel 2: Test-Time Adaptation Gain on Physical Image Bags
    tta = bench['tta_benchmark']
    methods = ['Source\n(No TTA)', 'TENT\n(Entropy Min)', 'HAT-MIL\n(Ours)']
    tta_acc = [tta['Source']['Accuracy'], tta['TENT']['Accuracy'], tta['HAT-MIL']['Accuracy']]
    tta_f1  = [tta['Source']['F1'], tta['TENT']['F1'], tta['HAT-MIL']['F1']]
    tta_auc = [tta['Source']['AUROC'], tta['TENT']['AUROC'], tta['HAT-MIL']['AUROC']]
    
    x2 = np.arange(len(methods))
    w2 = 0.25
    b1 = ax2.bar(x2 - w2, tta_acc, w2, label='Accuracy', color='#3182bd', edgecolor='black', linewidth=0.8)
    b2 = ax2.bar(x2, tta_auc, w2, label='AUROC', color='#e6550d', edgecolor='black', linewidth=0.8)
    b3 = ax2.bar(x2 + w2, tta_f1, w2, label='Macro-F1', color='#31a354', edgecolor='black', linewidth=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(methods, fontsize=8.5)
    ax2.set_ylabel("Score")
    ax2.set_ylim(0.65, 0.95)
    ax2.set_title("(b) Real Test-Time Adaptation Gain", fontweight='bold')
    ax2.legend(frameon=True, loc='lower right')
    
    # Annotate gain
    gain = (tta['HAT-MIL']['Accuracy'] - tta['Source']['Accuracy']) * 100
    ax2.annotate(f'+{gain:.1f}% Acc Gain', xy=(2 - w2, tta['HAT-MIL']['Accuracy']),
                 xytext=(1.2, 0.90),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
                 fontweight='bold', fontsize=9, color='#de2d26')

    plt.tight_layout()
    fig.savefig("experiments/figures/main_results.pdf")
    fig.savefig("Manuscript/main_results.pdf")
    plt.close()
    print("✓ Saved verified main_results.pdf")

if __name__ == '__main__':
    Path("experiments/figures").mkdir(parents=True, exist_ok=True)
    plot_dataset_eda()
    plot_ablation_results()
    plot_main_results()
    print("All publication figures successfully updated with real GPU metrics!")
