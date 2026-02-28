"""
Step: train_model
=================
Trains the CNN on the preprocessed CIFAR-10 training split.
Evaluates on the validation split per epoch.
Logs hyperparameters and metrics to MLflow.

Satisfies:
  R-TRAIN-5 — Train CNN; log loss/accuracy per epoch to MLflow
  R-MODEL-2 — Hyperparameters (lr, batch size, epochs, dropout) logged
  R-MODEL-3 — Metrics (train loss, val loss, val accuracy) logged
"""
import logging
from typing import Annotated, Tuple

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from zenml import step, ArtifactConfig
from zenml.client import Client

from src.models.cnn import CIFAR10CNN
from src.utils.helpers import load_split
from src.steps.preprocess import build_train_transform

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Custom PyTorch Dataset to apply transforms online
# ─────────────────────────────────────────────────────────────────
class TransformTensorDataset(torch.utils.data.Dataset):
    """
    Applies torchvision transforms to float32 tensors online.
    Important for RandomCrop/Flip randomness per epoch.
    """
    def __init__(self, images_tensor, labels_tensor, transform=None):
        self.images = images_tensor
        self.labels = labels_tensor
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ─────────────────────────────────────────────────────────────────
# ZenML Step
# ─────────────────────────────────────────────────────────────────
@step(experiment_tracker="mlflow_tracker")
def train_model(
    processed_data_path: str,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 0.001,
    dropout_rate: float = 0.5,
) -> Annotated[nn.Module, ArtifactConfig(name="trained_cnn", is_model_artifact=True)]:
    """
    Train CNN and log to MLflow.
    """
    # ── 1. Setup DataLoaders ─────────────────────────────────────────
    # We load the NORMALIZED numpy arrays (from preprocess step)
    X_train_np, y_train_np = load_split("train_normalized")
    X_val_np,   y_val_np   = load_split("val_normalized")

    X_train = torch.from_numpy(X_train_np)
    y_train = torch.from_numpy(y_train_np)
    X_val   = torch.from_numpy(X_val_np)
    y_val   = torch.from_numpy(y_val_np)

    # Train transform applies RandomCrop & RandomHorizontalFlip online
    # Note: X_train is ALREADY normalized, but RandomCrop works on tensors.
    # To avoid double-normalizing, we only use the spatial augmentations here.
    import torchvision.transforms as T
    train_aug = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
    ])

    train_ds = TransformTensorDataset(X_train, y_train, transform=train_aug)
    val_ds   = TensorDataset(X_val, y_val)  # No augmentation for val

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    # ── 2. Setup Device & Model ──────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    model = CIFAR10CNN(dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ── 3. MLflow Logging Setup ──────────────────────────────────────
    # Autolog covers optimizer state, but we manually log metrics to control naming
    mlflow.pytorch.autolog(log_models=False) # We'll log the model in register_model step
    
    mlflow.log_params({
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "dropout_rate": dropout_rate,
        "model": "CIFAR10CNN_3Blocks",
        "device": str(device)
    })

    # ── 4. Training Loop ─────────────────────────────────────────────
    logger.info("Starting training for %d epochs...", epochs)

    for epoch in range(1, epochs + 1):
        # ── Train phase ──
        model.train()
        train_loss_sum = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * batch_x.size(0)
            
        epoch_train_loss = train_loss_sum / len(train_loader.dataset)

        # ── Val phase ──
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss_sum += loss.item() * batch_x.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == batch_y.data).item()

        epoch_val_loss = val_loss_sum / len(val_loader.dataset)
        epoch_val_acc  = val_correct / len(val_loader.dataset)

        # ── Logging ──
        logger.info(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f}"
        )
        
        mlflow.log_metrics({
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "val_accuracy": epoch_val_acc
        }, step=epoch)

    logger.info("Training complete.")
    
    # Needs to be brought back to CPU for ZenML artifact storage
    return model.cpu()
