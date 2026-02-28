"""
Step: validate_data
===================
Validates the raw CIFAR-10 .npz files for shape, dtype, label range,
and absence of NaN values. Fails the pipeline if any check fails.

Satisfies:
  R-TRAIN-2 — shape checks, dtype, label range, no nulls; fail if checks fail
"""
import logging
from pathlib import Path

import numpy as np
from zenml import step

from src.utils.helpers import get_data_dir, NUM_CLASSES

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
EXPECTED_TRAIN_SAMPLES = 50_000
EXPECTED_TEST_SAMPLES  = 10_000
EXPECTED_IMAGE_SHAPE   = (32, 32, 3)   # H × W × C
EXPECTED_IMAGE_DTYPE   = np.uint8
EXPECTED_LABEL_DTYPE   = np.int64
LABEL_MIN, LABEL_MAX   = 0, NUM_CLASSES - 1   # 0 … 9


# ─────────────────────────────────────────────────────────────────
# Internal validation logic
# ─────────────────────────────────────────────────────────────────

def _validate_split(name: str, images: np.ndarray, labels: np.ndarray, n: int) -> None:
    """
    Run all assertions on a single dataset split.

    Raises:
        ValueError: if any validation check fails (causes pipeline to stop).
    """
    errors: list[str] = []

    # 1. Sample count
    if images.shape[0] != n:
        errors.append(f"[{name}] Expected {n} samples, got {images.shape[0]}")
    if labels.shape[0] != n:
        errors.append(f"[{name}] Expected {n} labels, got {labels.shape[0]}")

    # 2. Image spatial shape
    if images.shape[1:] != EXPECTED_IMAGE_SHAPE:
        errors.append(
            f"[{name}] Expected image shape {EXPECTED_IMAGE_SHAPE}, "
            f"got {images.shape[1:]}"
        )

    # 3. Dtype checks
    if images.dtype != EXPECTED_IMAGE_DTYPE:
        errors.append(
            f"[{name}] Expected image dtype {EXPECTED_IMAGE_DTYPE}, got {images.dtype}"
        )
    if labels.dtype != EXPECTED_LABEL_DTYPE:
        errors.append(
            f"[{name}] Expected label dtype {EXPECTED_LABEL_DTYPE}, got {labels.dtype}"
        )

    # 4. Label range [0, 9]
    if labels.min() < LABEL_MIN or labels.max() > LABEL_MAX:
        errors.append(
            f"[{name}] Label values out of range [{LABEL_MIN}, {LABEL_MAX}]: "
            f"min={labels.min()}, max={labels.max()}"
        )

    # 5. No NaN / null values (uint8 can't be NaN, but cast to float to be safe)
    if np.isnan(images.astype(np.float32)).any():
        errors.append(f"[{name}] NaN values found in images")

    if errors:
        msg = "\n".join(errors)
        logger.error("Validation FAILED:\n%s", msg)
        raise ValueError(f"Data validation failed:\n{msg}")

    logger.info("[%s] ✓ All checks passed (shape=%s, dtype=%s)", name, images.shape, images.dtype)


# ─────────────────────────────────────────────────────────────────
# ZenML step
# ─────────────────────────────────────────────────────────────────

@step
def validate_data(raw_data_path: str) -> str:
    """
    Validate CIFAR-10 .npz files for shape, dtype, label range, and NaN values.

    Args:
        raw_data_path: path to data/raw/ directory (output of ingest_data step)

    Returns:
        The same raw_data_path if validation passes (passed to next step).

    Raises:
        ValueError: if any validation check fails — pipeline halts.
    """
    raw_dir = Path(raw_data_path)

    for fname, expected_n in [
        ("cifar10_train.npz", EXPECTED_TRAIN_SAMPLES),
        ("cifar10_test.npz",  EXPECTED_TEST_SAMPLES),
    ]:
        path = raw_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{fname} not found at {path}. "
                "Run ingest_data step first."
            )

        logger.info("Loading %s for validation…", fname)
        data   = np.load(path)
        images = data["images"]
        labels = data["labels"]

        _validate_split(fname.replace(".npz", ""), images, labels, expected_n)

    logger.info("✅ All data validation checks passed.")
    return raw_data_path


# ─────────────────────────────────────────────────────────────────
# Standalone entry-point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    from src.utils.helpers import get_data_dir
    path = str(get_data_dir("raw"))
    result = validate_data.entrypoint(raw_data_path=path)
    print(f"Validation passed. Path: {result}")
