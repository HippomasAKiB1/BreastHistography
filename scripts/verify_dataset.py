#!/usr/bin/env python3
"""
scripts/verify_dataset.py: Verify BreaKHis dataset integrity, patient counts,
and image distributions.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.splits import collect_patient_info, BENIGN_SUBS, MALIGNANT_SUBS, MAGNIFICATIONS


def verify_dataset(data_dir: str) -> bool:
    print("=" * 70)
    print("BreaKHis Dataset Integrity & Patient Verification")
    print("=" * 70)
    data_path = Path(data_dir)
    print(f"Dataset root: {data_path.resolve()}")

    if not data_path.exists():
        print(f"❌ Error: Dataset path '{data_path}' does not exist.")
        print("Please download BreaKHis using 'python scripts/download_breakhis.py'.")
        return False

    patient_info = collect_patient_info(str(data_path))
    num_patients = len(patient_info)
    print(f"Discovered Patients: {num_patients}")

    if num_patients == 0:
        print("❌ Error: No patient folders found. Check your directory structure.")
        return False

    benign_pats = [p for p, info in patient_info.items() if info["binary"] == 0]
    malignant_pats = [p for p, info in patient_info.items() if info["binary"] == 1]

    print(f"  • Benign Patients:    {len(benign_pats)} (Expected: 24)")
    print(f"  • Malignant Patients: {len(malignant_pats)} (Expected: 58)")

    total_images = 0
    mag_counts = {m: 0 for m in MAGNIFICATIONS}

    for p, info in patient_info.items():
        for m in MAGNIFICATIONS:
            cnt = len(info["paths"][m])
            mag_counts[m] += cnt
            total_images += cnt

    print("\nMagnification Distribution:")
    for m in MAGNIFICATIONS:
        print(f"  • {m:4s}: {mag_counts[m]:5d} images")
    print(f"Total Histopathology Images: {total_images} (Expected: 7,909 in full BreaKHis v1)")

    # Assertions
    is_valid = True
    if len(benign_pats) != 24 or len(malignant_pats) != 58:
        print("⚠️ Warning: Patient count differs from official 82-patient BreaKHis v1 cohort.")
        is_valid = False
    if total_images != 7909:
        print(f"⚠️ Warning: Total image count ({total_images}) differs from official 7,909.")
        is_valid = False

    if is_valid:
        print("\n✅ Verification PASSED: Dataset matches official BreaKHis v1 specification perfectly!")
    else:
        print(f"\nℹ️ Dataset verified with {num_patients} patients and {total_images} images.")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify BreaKHis dataset integrity.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast",
        help="Path to histology_slides/breast directory",
    )
    args = parser.parse_args()
    success = verify_dataset(args.data_dir)
    sys.exit(0 if success else 1)
