"""
I/O helpers for loading configurations, JSON artifacts, and PyTorch checkpoints.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import torch


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Saves a dictionary to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(path: Union[str, Path]) -> Any:
    """Loads a JSON artifact file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Saves a data structure to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Loads PyTorch checkpoint safely with weights_only if supported.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(checkpoint_path, map_location=device)
