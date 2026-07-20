"""
Part 3: Baseline CNN Architecture
CS 898BA - Homework Three - Fish Classification

A custom CNN built from scratch (no pretrained backbones):
  - 3 conv blocks: 32 -> 64 -> 128 filters, ReLU activation, MaxPool after each
  - Flatten -> 1 fully connected hidden layer -> softmax output (via CrossEntropyLoss)
"""

import torch
import torch.nn as nn


class FishCNN(nn.Module):
    def __init__(self, num_classes, img_size=96, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # img_size / 2

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # img_size / 4

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # img_size / 8
        )

        flat_size = 128 * (img_size // 8) * (img_size // 8)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),  # raw logits; softmax applied via CrossEntropyLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = FishCNN(num_classes=6, img_size=96)
    print(model)
    dummy = torch.randn(2, 3, 96, 96)
    out = model(dummy)
    print("Output shape:", out.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
