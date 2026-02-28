"""
Step: export_model
==================
Exports the trained CNN to ONNX format and saves it in the models/ dir.
The returned directory path will be tracked by DVC in the final workflow.

Satisfies:
  R-TRAIN-8 — Export to ONNX or TorchScript; push model file to DVC/S3
  R-MODEL-4 — Model exported in ONNX for serving
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn
from zenml import step

from src.utils.helpers import get_models_dir

logger = logging.getLogger(__name__)

@step
def export_model(model: nn.Module) -> str:
    """
    Exports PyTorch model to ONNX format.
    """
    models_dir = get_models_dir()
    onnx_path = models_dir / "cifar10_cnn.onnx"
    
    # ── ONNX Export ──────────────────────────────────────────────────
    # We need a dummy input tensor of the correct shape (B, C, H, W)
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Needs to be brought back to CPU if it was on GPU
    model.cpu()
    model.eval()
    
    logger.info("Exporting model to ONNX: %s", onnx_path)
    
    torch.onnx.export(
        model,                      # model being run
        dummy_input,                # model input (or a tuple for multiple inputs)
        str(onnx_path),             # where to save the model
        export_params=True,         # store the trained parameter weights inside the model file
        opset_version=14,           # the ONNX version to export the model to
        do_constant_folding=True,   # whether to execute constant folding for optimization
        input_names=['input'],      # the model's input names
        output_names=['output'],    # the model's output names
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    logger.info("✅ Model exported successfully to ONNX format.")
    logger.info("Note: Run `dvc add models/` and `dvc push` to persist it.")
    
    return str(models_dir)
