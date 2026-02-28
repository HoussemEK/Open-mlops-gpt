"""
Step: preprocess
================
Normalises all splits using CIFAR-10 channel mean/std.
Applies data augmentation to the training split only.

Returns torchvision Transform objects (serialised as dicts of params)
to be consumed by the training step.

Satisfies:
  R-TRAIN-4 — normalise with CIFAR-10 mean/std; augmentation on train only
"""
import logging
from typing import Annotated

import numpy as np
import torch
import torchvision.transforms as T
from zenml import step

from src.utils.helpers import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    get_data_dir,
    load_split,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Transform builders
# ─────────────────────────────────────────────────────────────────

def build_train_transform() -> T.Compose:
    """
    Training transform:
      1. RandomCrop(32, padding=4)   — common CIFAR augmentation
      2. RandomHorizontalFlip()      — standard augmentation
      3. ToTensor()                  — [0, 255] uint8 → [0.0, 1.0] float32
      4. Normalize(mean, std)        — CIFAR-10 channel normalisation
    """
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def build_eval_transform() -> T.Compose:
    """
    Validation / test transform (no augmentation):
      1. ToTensor()
      2. Normalize(mean, std)
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


# ─────────────────────────────────────────────────────────────────
# Helper: pre-apply normalisation and save as float32 tensors
# ─────────────────────────────────────────────────────────────────

def _apply_and_save_normalized(
    split: str,
    images: np.ndarray,
    transform: T.Compose,
) -> None:
    """
    Applies ToTensor + Normalize to each image in the split and saves
    the result as a float32 .npz file in data/processed/.

    The augmentation transforms (RandomCrop, RandomHorizontalFlip) are
    intentionally NOT pre-applied here — they are applied online during
    DataLoader iteration in the training step.
    """
    from PIL import Image

    normalizer = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    processed = []
    for img in images:
        pil_img = Image.fromarray(img)
        tensor = normalizer(pil_img)
        processed.append(tensor.numpy())

    processed_arr = np.stack(processed, axis=0).astype(np.float32)  # (N, 3, 32, 32)
    labels_arr   = np.load(get_data_dir("processed") / f"{split}.npz")["labels"]

    out_path = get_data_dir("processed") / f"{split}_normalized.npz"
    np.savez_compressed(out_path, images=processed_arr, labels=labels_arr)
    logger.info("[%s] Normalised split saved → %s  shape=%s", split, out_path, processed_arr.shape)


# ─────────────────────────────────────────────────────────────────
# ZenML step
# ─────────────────────────────────────────────────────────────────

@step
def preprocess(processed_data_path: str) -> str:
    """
    Pre-apply normalisation to all splits (no augmentation stored on disk).
    Augmentation is applied online during training via the train transform.

    Args:
        processed_data_path: path to data/processed/ (from split_data step)

    Returns:
        The same processed_data_path (transforms are stateless constants).
    """
    logger.info("Preprocessing splits with CIFAR-10 normalisation …")
    logger.info("  mean = %s", CIFAR10_MEAN)
    logger.info("  std  = %s", CIFAR10_STD)

    for split in ("train", "val", "test"):
        images, _ = load_split(split)
        _apply_and_save_normalized(split, images, build_eval_transform())

    logger.info("✅ Normalised npz files written to data/processed/")
    logger.info(
        "Note: RandomCrop + RandomHorizontalFlip augmentation is applied "
        "online in the training DataLoader (build_train_transform())."
    )

    return processed_data_path


# ─────────────────────────────────────────────────────────────────
# Standalone entry-point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.utils.helpers import get_data_dir
    processed = str(get_data_dir("processed"))
    result = preprocess.entrypoint(processed_data_path=processed)
    print(f"Preprocessing done. Path: {result}")
