"""
Step: evaluate_model
====================
Evaluates the trained CNN on the held-out test split.
Computes comprehensive classification metrics and logs a confusion matrix 
plot directly to MLflow.

Satisfies:
  R-TRAIN-6 — Compute accuracy, precision, recall, F1, confusion matrix
"""
import logging
import os
import io

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from zenml import step

from src.utils.helpers import load_split, CIFAR10_CLASSES

logger = logging.getLogger(__name__)

@step(experiment_tracker="mlflow_tracker")
def evaluate_model(
    model: nn.Module,
    processed_data_path: str,
    batch_size: int = 128
) -> dict:
    """
    Evaluate the model on the test split and log metrics/artifacts to MLflow.
    """
    # ── 1. Load Test Data ────────────────────────────────────────────
    X_test_np, y_test_np = load_split("test_normalized")
    X_test = torch.from_numpy(X_test_np)
    y_test = torch.from_numpy(y_test_np)

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # ── 2. Run Inference ─────────────────────────────────────────────
    all_preds = []
    all_labels = []

    logger.info("Running evaluation on %d test images...", len(y_test_np))
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # ── 3. Compute Metrics ───────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    metrics = {
        "test_accuracy": acc,
        "test_precision_macro": precision,
        "test_recall_macro": recall,
        "test_f1_macro": f1
    }

    logger.info("Evaluation metrics: %s", metrics)
    mlflow.log_metrics(metrics)

    # ── 4. Generate & Log Confusion Matrix ───────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CIFAR10_CLASSES, 
        yticklabels=CIFAR10_CLASSES
    )
    plt.title("Confusion Matrix — Test Set")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save to a buffer and log to MLflow directly
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()
    
    # MLflow requires dumping to file before logging artifact usually, or using log_figure
    # In newer MLflow, log_figure is cleanly supported!
    # Workaround: write temp file then log.
    os.makedirs("/tmp", exist_ok=True)
    cm_path = "/tmp/confusion_matrix.png"
    with open(cm_path, "wb") as f:
        f.write(buf.getvalue())
        
    mlflow.log_artifact(cm_path, "evaluation_plots")
    logger.info("Logged confusion matrix plot to MLflow artifacts.")

    # Return metrics dictionary
    return metrics
