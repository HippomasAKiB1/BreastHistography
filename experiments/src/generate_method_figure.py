"""
generate_method_figure.py: Generates an updated, publication-quality architecture
flowchart for HAT-MIL and saves to Manuscript/fig_method.png and paper/figures/fig_method.png.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_method_figure():
    fig, ax = plt.subplots(figsize=(14, 5.2), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    # Color palette (modern academic)
    c_source_bg = '#edf2f7'
    c_target_bg = '#fff5f5'
    c_source_border = '#3182ce'
    c_target_border = '#e53e3e'
    
    # ── Source Domain Container ──
    rect_source = patches.FancyBboxPatch((0.2, 0.4), 6.4, 4.7, boxstyle="round,pad=0.2",
                                         facecolor=c_source_bg, edgecolor=c_source_border, linewidth=2, linestyle='--')
    ax.add_patch(rect_source)
    ax.text(3.4, 4.85, "SOURCE DOMAIN: Label-Aligned Pre-Training (BACH)", 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#2b6cb0')

    # Boxes in Source Domain
    # 1. Input Patches
    b1 = patches.FancyBboxPatch((0.5, 2.7), 1.6, 1.6, boxstyle="round,pad=0.1",
                                facecolor='#ebf8ff', edgecolor='#3182ce', linewidth=1.5)
    ax.add_patch(b1)
    ax.text(1.3, 3.7, "Tissue Patches", ha='center', va='center', fontsize=9.5, fontweight='bold')
    ax.text(1.3, 3.2, "256x256 px\nStride 128\nMax 32 / bag", ha='center', va='center', fontsize=8, color='#4a5568')

    # 2. Foundation Model
    b2 = patches.FancyBboxPatch((2.5, 2.7), 1.7, 1.6, boxstyle="round,pad=0.1",
                                facecolor='#faf5ff', edgecolor='#805ad5', linewidth=1.5)
    ax.add_patch(b2)
    ax.text(3.35, 3.7, "Foundation Model", ha='center', va='center', fontsize=9.5, fontweight='bold')
    ax.text(3.35, 3.2, "UNI (ViT-L/16)\n1024-D Features\nFrozen Weights", ha='center', va='center', fontsize=8, color='#4a5568')

    # 3. MIL Aggregator & Ontological Mapping
    b3 = patches.FancyBboxPatch((4.6, 2.7), 1.8, 1.6, boxstyle="round,pad=0.1",
                                facecolor='#f0fff4', edgecolor='#38a169', linewidth=1.5)
    ax.add_patch(b3)
    ax.text(5.5, 3.7, "Gated AB-MIL", ha='center', va='center', fontsize=9.5, fontweight='bold')
    ax.text(5.5, 3.2, "Attention Pooling\nLayerNorm\n2-Class Head", ha='center', va='center', fontsize=8, color='#4a5568')

    # Source Alignment banner at bottom of source
    b_align = patches.FancyBboxPatch((0.5, 0.7), 5.9, 1.5, boxstyle="round,pad=0.1",
                                     facecolor='#edfdfd', edgecolor='#319795', linewidth=1.5)
    ax.add_patch(b_align)
    ax.text(3.45, 1.8, "Ontological Label-Space Alignment", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#234e52')
    ax.text(3.45, 1.2, "Normal + Benign  -->  Class 0 (Non-Malignant)\nInSitu + Invasive  -->  Class 1 (Malignant)\n*Preserves trained classifier head across domains*",
            ha='center', va='center', fontsize=8.2, color='#285e61')

    # Arrows in Source
    ax.annotate('', xy=(2.5, 3.5), xytext=(2.1, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color='#4a5568'))
    ax.annotate('', xy=(4.6, 3.5), xytext=(4.2, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color='#4a5568'))
    ax.annotate('', xy=(3.45, 2.2), xytext=(3.45, 2.7), arrowprops=dict(arrowstyle="<->", lw=1.5, color='#319795', linestyle=':'))

    # ── Target Domain Container ──
    rect_target = patches.FancyBboxPatch((7.2, 0.4), 6.5, 4.7, boxstyle="round,pad=0.2",
                                         facecolor=c_target_bg, edgecolor=c_target_border, linewidth=2, linestyle='--')
    ax.add_patch(rect_target)
    ax.text(10.45, 4.85, "TARGET DOMAIN: Head-Aware Test-Time Adaptation (BreaKHis)", 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#c53030')

    # Boxes in Target Domain
    # 4. Target Input with Optical Shift
    b4 = patches.FancyBboxPatch((7.5, 2.7), 1.6, 1.6, boxstyle="round,pad=0.1",
                                facecolor='#fffaf0', edgecolor='#dd6b20', linewidth=1.5)
    ax.add_patch(b4)
    ax.text(8.3, 3.7, "Unlabeled Target", ha='center', va='center', fontsize=9.5, fontweight='bold')
    ax.text(8.3, 3.2, "BreaKHis Biopsies\n40X, 100X, 200X, 400X\nOptical Covariate Shift", ha='center', va='center', fontsize=8, color='#4a5568')

    # 5. HAT-MIL Core Engine
    b5 = patches.FancyBboxPatch((9.5, 1.8), 2.2, 2.5, boxstyle="round,pad=0.1",
                                facecolor='#fff5f5', edgecolor='#e53e3e', linewidth=2)
    ax.add_patch(b5)
    ax.text(10.6, 4.0, "HAT-MIL Adaptation", ha='center', va='center', fontsize=10, fontweight='bold', color='#9b2c2c')
    ax.text(10.6, 3.2, "1. Entropy Sharpening\n2. Prototype Alignment:\n   ||z - mu_y||^2\n3. Diversity Penalty:\n   prevents collapse\nUpdates: LayerNorm + Head",
            ha='center', va='center', fontsize=8, color='#4a5568')

    # 6. Final Inference & Diagnostic Prediction
    b6 = patches.FancyBboxPatch((12.1, 2.7), 1.4, 1.6, boxstyle="round,pad=0.1",
                                facecolor='#f0fff4', edgecolor='#38a169', linewidth=1.5)
    ax.add_patch(b6)
    ax.text(12.8, 3.7, "Prediction", ha='center', va='center', fontsize=9.5, fontweight='bold')
    ax.text(12.8, 3.2, "Benign / Malignant\nCalibrated ECE\nRobust AUROC", ha='center', va='center', fontsize=8, color='#4a5568')

    # Connection Between Source & Target
    ax.annotate('', xy=(7.5, 3.5), xytext=(6.4, 3.5),
                arrowprops=dict(arrowstyle="->", lw=2.5, color='#e53e3e'))
    ax.text(6.8, 3.8, "Transfer\nTrained Head", ha='center', va='center', fontsize=8, fontweight='bold', color='#e53e3e')

    # Arrows in Target
    ax.annotate('', xy=(9.5, 3.5), xytext=(9.1, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color='#4a5568'))
    ax.annotate('', xy=(12.1, 3.5), xytext=(11.7, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color='#4a5568'))

    # Save clean figure with no redundant in-image caption
    plt.tight_layout()
    fig.savefig("Manuscript/fig_method.png", bbox_inches='tight', dpi=300)
    fig.savefig("paper/figures/fig_method.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ Successfully generated and saved clean fig_method.png without footer text")

if __name__ == '__main__':
    draw_method_figure()
