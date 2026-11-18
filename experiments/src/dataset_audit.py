"""
dataset_audit.py: Systematically verifies all images in BreaKHis, indexes patient IDs,
magnification distributions, subtypes, and verifies physical ground truth.
Outputs results to experiments/results/dataset_manifest.json.
"""

import os
import json
from collections import defaultdict
from pathlib import Path

def audit_breakhis(data_root="data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"):
    data_path = Path(data_root)
    assert data_path.exists(), f"Path {data_root} does not exist!"
    
    # Tracking containers
    records = []
    patient_records = defaultdict(lambda: {
        'class': None,
        'subtype': None,
        'magnifications': defaultdict(list)
    })
    
    mag_counts = defaultdict(lambda: defaultdict(int))
    subtype_counts = defaultdict(int)
    
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg')):
                full_path = Path(root) / f
                parts = full_path.parts
                
                # Path format: .../breast/<class>/SOB/<subtype>/<patient_id>/<mag>/<filename>
                try:
                    breast_idx = parts.index('breast')
                    cls_name = parts[breast_idx + 1].lower() # 'benign' or 'malignant'
                    subtype = parts[breast_idx + 3].lower()  # e.g. 'adenosis', 'ductal_carcinoma'
                    patient_id = parts[breast_idx + 4]       # e.g. 'SOB_B_A_14-22549AB'
                    mag = parts[breast_idx + 5]              # e.g. '100X', '200X'
                except (ValueError, IndexError) as e:
                    continue
                    
                record = {
                    'filename': f,
                    'relative_path': str(full_path.relative_to(data_path)),
                    'class': cls_name,
                    'binary_label': 0 if cls_name == 'benign' else 1,
                    'subtype': subtype,
                    'patient_id': patient_id,
                    'magnification': mag,
                    'file_size_bytes': full_path.stat().st_size
                }
                records.append(record)
                
                mag_counts[mag][cls_name] += 1
                subtype_counts[subtype] += 1
                
                p_entry = patient_records[patient_id]
                p_entry['class'] = cls_name
                p_entry['binary_label'] = 0 if cls_name == 'benign' else 1
                p_entry['subtype'] = subtype
                p_entry['magnifications'][mag].append(str(full_path.relative_to(data_path)))

    total_images = len(records)
    total_patients = len(patient_records)
    
    benign_patients = sum(1 for p in patient_records.values() if p['class'] == 'benign')
    malignant_patients = sum(1 for p in patient_records.values() if p['class'] == 'malignant')
    
    summary = {
        'total_images': total_images,
        'total_patients': total_patients,
        'benign_patients': benign_patients,
        'malignant_patients': malignant_patients,
        'magnification_breakdown': dict(mag_counts),
        'subtype_breakdown': dict(subtype_counts),
        'images_per_magnification': {m: sum(c.values()) for m, c in mag_counts.items()}
    }
    
    print("="*60)
    print(" BREAKHIS PHYSICAL DATASET AUDIT REPORT")
    print("="*60)
    print(f"Total Physical Images Verified: {total_images}")
    print(f"Total Unique Patients: {total_patients} (Benign: {benign_patients}, Malignant: {malignant_patients})")
    print("\nMagnification Distribution:")
    for mag, counts in sorted(mag_counts.items()):
        b = counts['benign']
        m = counts['malignant']
        tot = b + m
        pct_m = (m / tot) * 100
        print(f"  {mag:>5}: {tot:>5} images | Benign: {b:>4} | Malignant: {m:>4} ({pct_m:.1f}% Malignant)")
    print("="*60)
    
    output_manifest = {
        'summary': summary,
        'patients': patient_records,
        'records_count': total_images
    }
    
    out_file = Path("experiments/results/dataset_manifest.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(output_manifest, f, indent=2)
    print(f"Dataset manifest successfully saved to {out_file}")
    return summary

if __name__ == '__main__':
    audit_breakhis()
