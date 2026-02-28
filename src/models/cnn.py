"""
CNN architecture for CIFAR-10 classification.
Satisfies R-MODEL-1:
- ≥ 2 conv blocks (Conv2d, BatchNorm, ReLU, MaxPool)
- Global average pooling
- Fully-connected head
- Configurable dropout rate
"""
import torch
import torch.nn as nn

class CIFAR10CNN(nn.Module):
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Global Average Pooling ensures output is exactly 1x1 spatially
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Head
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.global_pool(self.relu3(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
