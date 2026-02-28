"""
Step: ingest_data
=================
Downloads CIFAR-10 via torchvision, converts to NumPy arrays,
and saves them to data/raw/.

Satisfies:
  R-DATA-1  — CIFAR-10 raw data tracked by DVC
  R-TRAIN-1 — ingest step pulls data, validates checksum, outputs path
"""
import hashlib
import logging
from pathlib import Path

import numpy as np
import torchvision
import torchvision.transforms as transforms
from zenml import step

from src.utils.helpers import get_data_dir

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _torchvision_dataset_to_numpy(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Convert a torchvision CIFAR10 dataset to NumPy arrays."""
    images = np.array(dataset.data, dtype=np.uint8)  # (N, 32, 32, 3)
    labels = np.array(dataset.targets, dtype=np.int64)  # (N,)
    return images, labels


def _checksum(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ─────────────────────────────────────────────────────────────────
# ZenML step
# ─────────────────────────────────────────────────────────────────

@step
def ingest_data() -> str:
    """
    Download CIFAR-10 via torchvision and persist as numpy .npz files.

    Returns:
        Absolute path string to the data/raw/ directory.
    """
    raw_dir = get_data_dir("raw")
    train_path = raw_dir / "cifar10_train.npz"
    test_path  = raw_dir / "cifar10_test.npz"

    # ── Download (skips if already cached by torchvision) ──────────
    logger.info("Downloading CIFAR-10 training set…")
    train_ds = torchvision.datasets.CIFAR10(
        root=str(raw_dir / "_torchvision_cache"),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    logger.info("Downloading CIFAR-10 test set…")
    test_ds = torchvision.datasets.CIFAR10(
        root=str(raw_dir / "_torchvision_cache"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    # ── Convert to NumPy ────────────────────────────────────────────
    train_images, train_labels = _torchvision_dataset_to_numpy(train_ds)
    test_images,  test_labels  = _torchvision_dataset_to_numpy(test_ds)

    logger.info("Train — images: %s  labels: %s", train_images.shape, train_labels.shape)
    logger.info("Test  — images: %s  labels: %s", test_images.shape,  test_labels.shape)

    # ── Save as compressed .npz ─────────────────────────────────────
    np.savez_compressed(train_path, images=train_images, labels=train_labels)
    np.savez_compressed(test_path,  images=test_images,  labels=test_labels)
    logger.info("Saved → %s", train_path)
    logger.info("Saved → %s", test_path)

    # ── Checksum validation (R-TRAIN-1) ──────────────────────────────
    for p in (train_path, test_path):
        chk = _checksum(p)
        logger.info("SHA-256 [%s]: %s", p.name, chk)

    return str(raw_dir)


# ─────────────────────────────────────────────────────────────────
# Standalone entry-point (run without ZenML for quick testing)
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raw_dir_path = ingest_data.entrypoint()
    print(f"Data saved to: {raw_dir_path}")
