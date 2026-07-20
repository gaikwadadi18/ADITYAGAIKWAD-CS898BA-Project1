"""
Part 4 (cont.): Train the final "optimized" model using the best hyperparameter
config found during the random search, for the same number of epochs as the
baseline (10) so the two are fairly comparable in Part 5.
"""

import json
import time
import torch
import torch.nn as nn

from src.data_pipeline import get_dataloaders
from src.model import FishCNN
from src.train_baseline import run_epoch, DEVICE, EPOCHS

MODEL_OUT = "/home/claude/fish_project/outputs/models/optimized_cnn.pt"
HISTORY_OUT = "/home/claude/fish_project/outputs/reports/optimized_history.json"
BEST_CONFIG_PATH = "/home/claude/fish_project/outputs/reports/best_config.json"


def train_optimized():
    with open(BEST_CONFIG_PATH) as f:
        config = json.load(f)
    print("Training optimized model with config:", config)

    train_loader, val_loader, test_loader, class_weights, info = get_dataloaders(batch_size=config["batch_size"])

    model = FishCNN(num_classes=len(info["classes"]), dropout=config["dropout"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))  # class-weighted, addresses imbalance
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

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
    print(f"Saved optimized model weights to {MODEL_OUT}")

    with open(HISTORY_OUT, "w") as f:
        json.dump({"config": config, "history": history}, f, indent=2)
    print(f"Saved optimized training history to {HISTORY_OUT}")

    return model, history, config, info


if __name__ == "__main__":
    train_optimized()
