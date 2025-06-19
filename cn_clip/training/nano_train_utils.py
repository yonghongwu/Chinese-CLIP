import os, sys
os.chdir('/database/wuyonghuang/WSA')
import torch
import torch.nn.functional as F

from torch import nn


# Assume these U-Net encoders are pre-defined or imported
# They should output a feature vector (e.g., after pooling and flattening)
# For simplicity, let's assume they output a fixed-size feature vector, e.g., 512-dim.


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量（Total parameters）: {total:,}")
    print(f"可训练参数量（Trainable parameters）: {trainable:,}")
    print(f"冻结参数量（Frozen parameters）: {total - trainable:,}")
    print(f"训练参数比例为:{trainable / total:.2f}")


class UNet3DEncoder(nn.Module):
    def __init__(self, in_channels, out_features=512):
        super().__init__()
        # ... (Full 3D U-Net encoder architecture: conv blocks, downsampling)
        # Example:
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        # ... more layers ...
        self.bottleneck_conv = nn.Conv3d(64, 128, kernel_size=3, padding=1) # Example output before pooling
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, out_features) # Adjust 128 based on actual bottleneck channels

    def forward(self, x):
        # x shape: (B, C_in, D, H, W) e.g., (B, 2, 64, 64, 64) for MRI
        # ... (Pass through U-Net encoder layers)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # ...
        x = F.relu(self.bottleneck_conv(x))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return x


class UNet2DEncoder(nn.Module):
    def __init__(self, in_channels, out_features=512):
        super().__init__()
        # ... (Full 2D U-Net encoder architecture: conv blocks, downsampling)
        # Example:
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        # ... more layers ...
        self.bottleneck_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Example output before pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, out_features) # Adjust 128 based on actual bottleneck channels

    def forward(self, x):
        # x shape: (B, C_in, H, W) e.g., (B, 2, 256, 256) for US
        # ... (Pass through U-Net encoder layers)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # ...
        x = F.relu(self.bottleneck_conv(x))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return x

