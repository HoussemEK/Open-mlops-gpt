from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from zenml import step

from src.models.cnn import CIFAR10CNN

def apply_drift(images: np.ndarray) -> np.ndarray:
    """Apply severe jitter and noise to simulate data drift."""
    # Convert numpy (N, H, W, C) back to PIL/Tensors for torchvision transforms
    # We'll batch process for efficiency. This isn't strictly realistic but works for the demo.
    
    # Simple explicit operations to corrupt the numpy arrays simulating "bad camera sensors" or "night mode" drift
    corrupted = images.astype(np.float32)
    
    # 1. Add Gaussian noise
    noise = np.random.normal(loc=0.0, scale=40.0, size=corrupted.shape)
    corrupted += noise
    
    # 2. Adjust contrast (scale pixel intensities towards 128)
    corrupted = (corrupted - 128.0) * 0.5 + 128.0
    
    # Clip back to valid image range
    return np.clip(corrupted, 0, 255).astype(np.uint8)

@step
def collect_inference_data(
    test_data_path: str,
    simulate_drift: bool = False,
    batch_size: int = 128,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulates collecting production inference data.
    
    Args:
        test_data_path: Path to the clean test.npz dataset.
        simulate_drift: If True, applies significant corruption to simulate drift.
        batch_size: Batch size for generating predictions.
        
    Returns:
        reference_df: A dataframe of clean test data features and labels (as the baseline).
        current_df: A dataframe of current inference features and production predictions.
    """
    # 1. Load the clean test dataset to act as both our reference and our source of current traffic
    data = np.load(test_data_path)
    images = data["images"]
    labels = data["labels"]
    
    # To keep things light for evidently report, we'll only take a subset (e.g. 500 samples)
    sample_idx = np.random.choice(len(images), 500, replace=False)
    
    images_ref = images[sample_idx]
    labels_ref = labels[sample_idx]
    
    images_cur = images_ref.copy()
    
    # 2. Simulate drift if requested
    if simulate_drift:
        images_cur = apply_drift(images_cur)
    
    # 3. Create evidently requires tabular data by default for computer vision (we extract basic image features)
    # We will compute basic statistical features (mean intensity per channel) for Evidently Tabular Drift
    def extract_features(img_array):
        # img_array is (N, 32, 32, 3)
        # return (N, 3) mean R, G, B
        means = img_array.mean(axis=(1, 2))
        stds = img_array.std(axis=(1, 2))
        # Add basic brightness/contrast features
        features = np.hstack([means, stds])
        return pd.DataFrame(features, columns=["mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B"])
        
    reference_df = extract_features(images_ref)
    current_df = extract_features(images_cur)
    
    reference_df["target"] = labels_ref
    
    # In a real pipeline, current_df targets wouldn't be known, but for demo we can mock predictions 
    # taking the real labels (if drift is False) or randomly misclassifying (if drift is True)
    if simulate_drift:
        # Drift messes up predictions
        current_df["prediction"] = np.random.randint(0, 10, size=len(images_cur))
    else:
        # No drift, predictions match closely
        current_df["prediction"] = labels_ref
    
    return reference_df, current_df
