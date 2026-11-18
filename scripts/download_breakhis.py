#!/usr/bin/env python3
"""
scripts/download_breakhis.py: Download and setup instructions for the BreaKHis 400X dataset.
"""

import os
import sys
import argparse
from pathlib import Path

DATASET_URLS = {
    "kaggle": "kaggle datasets download -d aminaso/breakhis-400x -p data/",
    "official_span": "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/",
}


def print_instructions(target_dir: str):
    target_path = Path(target_dir).resolve()
    print("=" * 75)
    print("BreaKHis Dataset Download & Setup Guide")
    print("=" * 75)
    print(f"\nTarget extraction directory:\n  {target_path}\n")
    print("Option 1: Using Kaggle CLI (Recommended)")
    print("  1. Ensure you have kaggle installed and configured: pip install kaggle")
    print("  2. Run:")
    print("     kaggle datasets download -d aminaso/breakhis-400x -p data/")
    print("  3. Unzip the downloaded archive:")
    print("     unzip data/breakhis-400x.zip -d data/BreaKHis_v1/")
    print("\nOption 2: Official UFPR Laboratory Portal")
    print(f"  Visit: {DATASET_URLS['official_span']}")
    print("  Download the BreaKHis_v1.tar.gz archive and extract to data/BreaKHis_v1/")
    print("\nExpected directory layout after extraction:")
    print("  data/")
    print("  └── BreaKHis_v1/")
    print("      └── BreaKHis_v1/")
    print("          └── histology_slides/")
    print("              └── breast/")
    print("                  ├── benign/")
    print("                  │   └── SOB/")
    print("                  │       ├── adenosis/")
    print("                  │       ├── fibroadenoma/")
    print("                  │       ├── phyllodes_tumor/")
    print("                  │       └── tubular_adenoma/")
    print("                  └── malignant/")
    print("                      └── SOB/")
    print("                          ├── ductal_carcinoma/")
    print("                          ├── lobular_carcinoma/")
    print("                          ├── mucinous_carcinoma/")
    print("                          └── papillary_carcinoma/")
    print("\nAfter setting up the dataset, run verification:")
    print("  python scripts/verify_dataset.py")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & Setup Guide for BreaKHis")
    parser.add_argument("--target_dir", type=str, default="data/BreaKHis_v1", help="Target dataset directory")
    args = parser.parse_args()
    print_instructions(args.target_dir)
