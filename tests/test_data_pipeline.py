"""
Unit tests for data transforms, stain normalisation, and patient split integrity.
"""

import pytest
import numpy as np
from PIL import Image
import torch

from src.data import (
    macenko_norm,
    get_train_transforms,
    get_eval_transforms,
    get_fixed_train_val_test_split,
    get_stratified_kfold_splits,
)


def test_macenko_norm():
    # Synthetic RGB image
    arr = np.random.randint(50, 200, size=(100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    norm_img = macenko_norm(img)
    assert norm_img.size == (100, 100)
    assert norm_img.mode == "RGB"


def test_eval_transforms():
    tf = get_eval_transforms(img_size=224)
    img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    tensor = tf(img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_patient_split_no_leakage():
    # Generate 40 synthetic patients
    patients = {}
    for i in range(40):
        pid = f"SOB_B_A-14-{i:04d}"
        patients[pid] = {
            "binary": 0 if i < 15 else 1,
            "subtype": 0 if i < 15 else 4,
            "paths": {"40X": [], "100X": [], "200X": [], "400X": []},
        }

    train, val, test = get_fixed_train_val_test_split(patients, seed=42)

    # Invariants: No patient overlap across splits
    s_train = set(train)
    s_val = set(val)
    s_test = set(test)

    assert len(s_train.intersection(s_val)) == 0, "Patient leakage between train and val!"
    assert len(s_train.intersection(s_test)) == 0, "Patient leakage between train and test!"
    assert len(s_val.intersection(s_test)) == 0, "Patient leakage between val and test!"
    assert len(s_train) + len(s_val) + len(s_test) == 40


def test_stratified_kfold_splits():
    patients = {f"P_{i}": {"binary": 0 if i < 20 else 1} for i in range(50)}
    folds = get_stratified_kfold_splits(patients, n_splits=5, seed=42)
    assert len(folds) == 5
    for train_p, val_p in folds:
        s_tr = set(train_p)
        s_val = set(val_p)
        assert len(s_tr.intersection(s_val)) == 0
        assert len(s_tr) + len(s_val) == 50
