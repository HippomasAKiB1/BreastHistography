"""
Deterministic seed configuration for reproducible experiments.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_cuda: bool = True) -> None:
    """
    Sets deterministic seeds across Python, NumPy, PyTorch CPU, and CUDA.

    Args:
        seed: Random seed integer.
        deterministic_cuda: If True, sets torch.backends.cudnn.deterministic = True
            and benchmark = False.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
