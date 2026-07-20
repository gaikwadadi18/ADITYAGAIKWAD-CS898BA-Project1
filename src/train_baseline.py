"""
Part 3: Baseline CNN Training
CS 898BA - Homework Three - Fish Classification

Trains the baseline FishCNN with standard hyperparameters:
  - Optimizer: Adam
  - Learning rate: 0.001
  - Batch size: 32
  - Epochs: 10 (fixed)

Saves model weights and training/validation loss & accuracy curves.
"""

import time
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.data_pipeline import get_dataloaders
from src.model import FishCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
LR = 0.001
BATCH_SIZE = 32

MODEL_OUT = "/home/claude/fish_project/outputs/models/baseline_cnn.pt"
PLOT_OUT = "/home/claude/fish_project/outputs/plots/baseline_curves.png"
HISTORY_OUT = "/home/claude/fish_project/outputs/reports/baseline_history.json"


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            out = model(xb)
            loss = criterion(out, yb)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_correct += (out.argmax(1) == yb).sum().item()
            total_samples += xb.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_baseline():
    train_loader, val_loader, test_loader, class_weights, info = get_dataloaders(batch_size=BATCH_SIZE)

    model = FishCNN(num_classes=len(info["classes"])).to(DEVICE)
    criterion = nn.CrossEntropyLoss()  # baseline: no class weighting yet, kept "standard"
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"Training baseline on device: {DEVICE}")
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    elapsed = time.time() - t0
    print(f"Training finished in {elapsed:.1f}s")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Saved model weights to {MODEL_OUT}")

    with open(HISTORY_OUT, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {HISTORY_OUT}")

    # Plot loss & accuracy curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs_range = range(1, EPOCHS + 1)

    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs_range, history["val_loss"], label="Val Loss", marker="o")
    axes[0].set_title("Baseline: Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(epochs_range, history["val_acc"], label="Val Acc", marker="o")
    axes[1].set_title("Baseline: Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=150)
    print(f"Saved loss/accuracy curves to {PLOT_OUT}")

    return model, history, info


if __name__ == "__main__":
    train_baseline()
