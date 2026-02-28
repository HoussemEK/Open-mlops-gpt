"""
Shared utility helpers for the OpenMLOps Challenge.
"""
import os
import json
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────
# CIFAR-10 normalisation constants
# ─────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

NUM_CLASSES = 10


# ─────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────
def get_project_root() -> Path:
    """Return the absolute project root directory."""
    return Path(__file__).resolve().parents[2]


def get_data_dir(subdir: str = "") -> Path:
    """Return path to data/ or a subdirectory of data/."""
    p = get_project_root() / "data" / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_models_dir() -> Path:
    """Return path to models/ directory."""
    p = get_project_root() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_reports_dir() -> Path:
    """Return path to reports/ directory."""
    p = get_project_root() / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────
def get_mlflow_tracking_uri() -> str:
    """Read MLFLOW_TRACKING_URI from env with fallback."""
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def use_drifted_data() -> bool:
    """Check whether the monitoring pipeline should use the drifted dataset."""
    return os.environ.get("USE_DRIFTED_DATA", "false").lower() == "true"


# ─────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────
def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a preprocessed split (train / val / test) from data/processed/.

    Args:
        split: one of 'train', 'val', 'test'

    Returns:
        Tuple of (images, labels) as numpy arrays.
    """
    path = get_data_dir("processed") / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Split '{split}' not found at {path}. "
            "Run ingest_data → validate_data → split_data first."
        )
    data = np.load(path)
    return data["images"], data["labels"]


def save_split(images: np.ndarray, labels: np.ndarray, split: str) -> Path:
    """
    Save a dataset split to data/processed/{split}.npz.

    Returns the file path.
    """
    out_path = get_data_dir("processed") / f"{split}.npz"
    np.savez_compressed(out_path, images=images, labels=labels)
    return out_path


# ─────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────
def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
