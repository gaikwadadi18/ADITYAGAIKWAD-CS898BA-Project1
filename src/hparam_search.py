"""
Part 4: Hyperparameter Optimization
CS 898BA - Homework Three - Fish Classification

Tuning strategy: Random Search over 3 hyperparameters
  - Learning rate: {0.01, 0.001, 0.0001}
  - Batch size: {32, 64}
  - Regularization: dropout {0.3, 0.5} AND L2 weight decay {0.0, 1e-4}
    (varied as separate axes, so a config might combine e.g. dropout=0.3 with weight_decay=1e-4)

We randomly sample 6 distinct configs from the full grid (3*2*2*2=24 possible combos)
and train each for 6 epochs (kept short due to CPU-only single-core sandbox).
The best config is selected by lowest validation loss, then retrained for longer
as the final "optimized" model in a separate step.
"""

import json
import random
import itertools
import time

import torch
import torch.nn as nn

from src.data_pipeline import get_dataloaders
from src.model import FishCNN
from src.train_baseline import run_epoch, DEVICE

SEARCH_EPOCHS = 6
N_TRIALS = 6
SEED = 42

RESULTS_OUT = "/home/claude/fish_project/outputs/reports/hparam_search_results.json"

random.seed(SEED)

LR_GRID = [0.01, 0.001, 0.0001]
BATCH_GRID = [32, 64]
DROPOUT_GRID = [0.3, 0.5]
WEIGHT_DECAY_GRID = [0.0, 1e-4]


def sample_configs(n_trials):
    full_grid = list(itertools.product(LR_GRID, BATCH_GRID, DROPOUT_GRID, WEIGHT_DECAY_GRID))
    random.shuffle(full_grid)
    chosen = full_grid[:n_trials]
    configs = []
    for lr, bs, dropout, wd in chosen:
        configs.append({"lr": lr, "batch_size": bs, "dropout": dropout, "weight_decay": wd})
    return configs


def train_one_config(config, num_classes, epochs=SEARCH_EPOCHS):
    train_loader, val_loader, _, class_weights, info = get_dataloaders(batch_size=config["batch_size"])

    model = FishCNN(num_classes=num_classes, dropout=config["dropout"]).to(DEVICE)
    # Use class-weighted loss during tuning to address imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

    return model, history


def run_search():
    _, _, _, _, info = get_dataloaders()
    num_classes = len(info["classes"])

    configs = sample_configs(N_TRIALS)
    results = []

    print(f"Running random search: {N_TRIALS} trials x {SEARCH_EPOCHS} epochs on {DEVICE}\n")

    best_val_loss = float("inf")
    best_config = None
    best_model_state = None

    for i, config in enumerate(configs, 1):
        t0 = time.time()
        print(f"Trial {i}/{N_TRIALS}: {config}")
        model, history = train_one_config(config, num_classes)
        elapsed = time.time() - t0

        final_val_loss = history["val_loss"][-1]
        final_val_acc = history["val_acc"][-1]
        best_epoch_val_loss = min(history["val_loss"])

        print(f"  -> final val_loss={final_val_loss:.4f} val_acc={final_val_acc:.4f} "
              f"(best val_loss over run: {best_epoch_val_loss:.4f}) [{elapsed:.1f}s]\n")

        results.append({
            "config": config,
            "history": history,
            "final_val_loss": final_val_loss,
            "final_val_acc": final_val_acc,
            "best_val_loss": best_epoch_val_loss,
            "time_sec": elapsed,
        })

        if best_epoch_val_loss < best_val_loss:
            best_val_loss = best_epoch_val_loss
            best_config = config
            best_model_state = model.state_dict()

    print("=" * 60)
    print("Best config (by validation loss):", best_config)
    print(f"Best validation loss achieved: {best_val_loss:.4f}")

    with open(RESULTS_OUT, "w") as f:
        json.dump({"results": results, "best_config": best_config, "best_val_loss": best_val_loss}, f, indent=2)
    print(f"Saved full search results to {RESULTS_OUT}")

    return best_config, results


def run_single_trial(trial_idx):
    """Run one trial from the fixed config list and append its result to disk."""
    import os
    _, _, _, _, info = get_dataloaders()
    num_classes = len(info["classes"])
    configs = sample_configs(N_TRIALS)

    if trial_idx >= len(configs):
        print("No more trials.")
        return

    config = configs[trial_idx]
    print(f"Trial {trial_idx+1}/{len(configs)}: {config}")
    t0 = time.time()
    model, history = train_one_config(config, num_classes)
    elapsed = time.time() - t0

    final_val_loss = history["val_loss"][-1]
    final_val_acc = history["val_acc"][-1]
    best_epoch_val_loss = min(history["val_loss"])
    print(f"  -> final val_loss={final_val_loss:.4f} val_acc={final_val_acc:.4f} "
          f"(best val_loss over run: {best_epoch_val_loss:.4f}) [{elapsed:.1f}s]")

    result = {
        "trial_idx": trial_idx,
        "config": config,
        "history": history,
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc,
        "best_val_loss": best_epoch_val_loss,
        "time_sec": elapsed,
    }

    # Append to results file
    if os.path.exists(RESULTS_OUT):
        with open(RESULTS_OUT, "r") as f:
            data = json.load(f)
    else:
        data = {"results": []}

    data["results"].append(result)
    with open(RESULTS_OUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Appended result to {RESULTS_OUT}")

    # Save model weights for this trial too (small dataset, cheap to keep all)
    import torch as _torch
    _torch.save(model.state_dict(), f"/home/claude/fish_project/outputs/models/trial_{trial_idx}.pt")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        run_single_trial(int(sys.argv[2]))
    else:
        run_search()
