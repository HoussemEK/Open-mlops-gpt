"""
Step: split_data
================
Stratified train / val / test split of the raw CIFAR-10 data.
Saves each split as a compressed .npz file in data/processed/.

Satisfies:
  R-TRAIN-3 — stratified split: train 70%, val 15%, test 15%; saves splits
  R-DATA-4  — processed splits stored as separate DVC-tracked artefacts
"""
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from zenml import step

from src.utils.helpers import get_data_dir, save_split

logger = logging.getLogger(__name__)

# Split ratios per the plan: 70 / 15 / 15
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15   # of total → 0.15 / 0.30 of remaining after train split
TEST_RATIO  = 0.15   # of total → 0.50 of remaining after train split
RANDOM_SEED = 42


@step
def split_data(raw_data_path: str) -> str:
    """
    Stratified split of CIFAR-10 into train / val / test sets.

    Uses only the 50k training images (test set is held-out evaluation).
    Splits: 70 % train, 15 % val, 15 % test.

    Args:
        raw_data_path: path to data/raw/ (output of validate_data step)

    Returns:
        Path to data/processed/ directory.
    """
    raw_dir = Path(raw_data_path)
    train_file = raw_dir / "cifar10_train.npz"

    logger.info("Loading raw training data from %s …", train_file)
    data   = np.load(train_file)
    images = data["images"]   # (50000, 32, 32, 3), uint8
    labels = data["labels"]   # (50000,),           int64

    # ── Step 1: split off test set (15 %) ──────────────────────────
    # val_ratio_of_remaining = TEST_RATIO / (1 - TRAIN_RATIO) = 0.15 / 0.30 = 0.50
    val_ratio_of_remaining = TEST_RATIO / (1.0 - TRAIN_RATIO)

    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels,
        test_size=(1.0 - TRAIN_RATIO),
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    # ── Step 2: split remaining into val / test (50 / 50) ──────────
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_of_remaining,
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )

    logger.info("Train : %s images, %s labels", X_train.shape, y_train.shape)
    logger.info("Val   : %s images, %s labels", X_val.shape,   y_val.shape)
    logger.info("Test  : %s images, %s labels", X_test.shape,  y_test.shape)

    # ── Verify class balance ────────────────────────────────────────
    for split_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        logger.info("Class distribution [%s]: %s", split_name, dist)

    # ── Save splits ─────────────────────────────────────────────────
    processed_dir = get_data_dir("processed")
    save_split(X_train, y_train, "train")
    save_split(X_val,   y_val,   "val")
    save_split(X_test,  y_test,  "test")
    logger.info("Splits saved to %s", processed_dir)

    return str(processed_dir)


# ─────────────────────────────────────────────────────────────────
# Standalone entry-point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.utils.helpers import get_data_dir
    raw = str(get_data_dir("raw"))
    out = split_data.entrypoint(raw_data_path=raw)
    print(f"Splits saved to: {out}")
