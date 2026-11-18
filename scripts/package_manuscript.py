#!/usr/bin/env python3
"""
scripts/package_manuscript.py: Package complete publication-ready manuscript bundle
into a 1-click Overleaf zip archive.
"""

import sys
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MANUSCRIPT_DIR = ROOT / "Manuscript"
PAPER_DIR = ROOT / "paper"
ZIP_PATH = PAPER_DIR / "breasthistography_manuscript.zip"

REQUIRED_FILES = [
    "main.tex",
    "references.bib",
    "IEEEtran.cls",
    "IEEEtran.bst",
]

FIGURE_FILES = [
    "dataset_eda.pdf",
    "fig_method.png",
    "main_results.pdf",
    "ablation_results.pdf",
]


def sync_and_package():
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = PAPER_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Packaging BreastHistography Manuscript Bundle")
    print("=" * 60)

    # Copy primary source files to paper/
    for fname in REQUIRED_FILES:
        src = MANUSCRIPT_DIR / fname
        dst = PAPER_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ Synced {fname}")
        else:
            print(f"  ⚠️ Warning: Source {src} not found!")

    # Copy figure files to paper/figures/ and paper/ (for flexible LaTeX paths)
    for figname in FIGURE_FILES:
        src = MANUSCRIPT_DIR / figname
        if not src.exists():
            src = ROOT / "paper" / "figures" / figname
        if not src.exists():
            src = ROOT / "experiments" / "figures" / figname

        if src.exists():
            shutil.copy2(src, figures_dir / figname)
            shutil.copy2(src, PAPER_DIR / figname)
            print(f"  ✓ Synced figure {figname}")
        else:
            print(f"  ⚠️ Figure {figname} not found!")

    # Build zip file
    print(f"\nCreating archive: {ZIP_PATH.name}...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add root latex and bib files
        for fname in REQUIRED_FILES:
            fpath = PAPER_DIR / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)

        # Add figures in root and in figures/ folder
        for figname in FIGURE_FILES:
            fpath = figures_dir / figname
            if fpath.exists():
                zf.write(fpath, arcname=f"figures/{figname}")
                zf.write(fpath, arcname=figname)

    zip_size_kb = ZIP_PATH.stat().st_size / 1024.0
    print(f"✅ Archive built successfully: {ZIP_PATH} ({zip_size_kb:.1f} KB)")
    print("Ready for 1-click import into Overleaf or IEEE conference submission portals.")


if __name__ == "__main__":
    sync_and_package()
