"""
Part 5: Evaluation and Analysis
CS 898BA - Homework Three - Fish Classification

- Evaluates baseline and optimized models on the held-out TEST set
- Computes Accuracy, Precision, Recall, F1-Score (per-class + macro/weighted avg)
- Generates a confusion matrix for the optimized model
- Produces a combined figure: baseline vs optimized loss/acc curves + confusion matrix
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.data_pipeline import get_dataloaders
from src.model import FishCNN
from src.train_baseline import DEVICE

BASELINE_WEIGHTS = "/home/claude/fish_project/outputs/models/baseline_cnn.pt"
OPTIMIZED_WEIGHTS = "/home/claude/fish_project/outputs/models/optimized_cnn.pt"
BEST_CONFIG_PATH = "/home/claude/fish_project/outputs/reports/best_config.json"
BASELINE_HISTORY = "/home/claude/fish_project/outputs/reports/baseline_history.json"
OPTIMIZED_HISTORY = "/home/claude/fish_project/outputs/reports/optimized_history.json"

REPORTS_OUT = "/home/claude/fish_project/outputs/reports/test_classification_reports.json"
COMBINED_PLOT_OUT = "/home/claude/fish_project/outputs/plots/comparison_grid.png"


@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        out = model(xb)
        preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())
    return np.array(all_labels), np.array(all_preds)


def evaluate():
    with open(BEST_CONFIG_PATH) as f:
        best_config = json.load(f)

    # Baseline uses batch_size=32 loaders; optimized uses its own batch_size for train,
    # but test set evaluation batch size doesn't affect metrics -- use a fixed one.
    _, _, test_loader, _, info = get_dataloaders(batch_size=32)
    classes = info["classes"]
    num_classes = len(classes)

    # --- Load baseline ---
    baseline_model = FishCNN(num_classes=num_classes, dropout=0.5).to(DEVICE)  # baseline default dropout
    baseline_model.load_state_dict(torch.load(BASELINE_WEIGHTS, map_location=DEVICE))

    # --- Load optimized ---
    optimized_model = FishCNN(num_classes=num_classes, dropout=best_config["dropout"]).to(DEVICE)
    optimized_model.load_state_dict(torch.load(OPTIMIZED_WEIGHTS, map_location=DEVICE))

    results = {}
    for name, model in [("baseline", baseline_model), ("optimized", optimized_model)]:
        y_true, y_pred = get_predictions(model, test_loader)
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
        results[name] = {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "report": report,
        }

        print(f"\n{'='*20} {name.upper()} MODEL - TEST SET {'='*20}")
        print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
        print(f"Overall Accuracy: {report['accuracy']:.4f}")

    with open(REPORTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved classification reports to {REPORTS_OUT}")

    return results, classes


def plot_comparison_grid(results, classes):
    with open(BASELINE_HISTORY) as f:
        baseline_hist = json.load(f)
    with open(OPTIMIZED_HISTORY) as f:
        optimized_hist_data = json.load(f)
    optimized_hist = optimized_hist_data["history"]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)

    epochs_b = range(1, len(baseline_hist["train_loss"]) + 1)
    epochs_o = range(1, len(optimized_hist["train_loss"]) + 1)

    # Top-left: Loss curves (both models)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs_b, baseline_hist["train_loss"], label="Baseline Train", linestyle="--", color="tab:blue")
    ax1.plot(epochs_b, baseline_hist["val_loss"], label="Baseline Val", color="tab:blue")
    ax1.plot(epochs_o, optimized_hist["train_loss"], label="Optimized Train", linestyle="--", color="tab:orange")
    ax1.plot(epochs_o, optimized_hist["val_loss"], label="Optimized Val", color="tab:orange")
    ax1.set_title("Loss: Baseline vs Optimized")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Top-right: Accuracy curves (both models)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_b, baseline_hist["train_acc"], label="Baseline Train", linestyle="--", color="tab:blue")
    ax2.plot(epochs_b, baseline_hist["val_acc"], label="Baseline Val", color="tab:blue")
    ax2.plot(epochs_o, optimized_hist["train_acc"], label="Optimized Train", linestyle="--", color="tab:orange")
    ax2.plot(epochs_o, optimized_hist["val_acc"], label="Optimized Val", color="tab:orange")
    ax2.set_title("Accuracy: Baseline vs Optimized")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Bottom: Confusion matrix (optimized model), spanning both columns
    ax3 = fig.add_subplot(gs[1, :])
    y_true = results["optimized"]["y_true"]
    y_pred = results["optimized"]["y_pred"]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax3)
    ax3.set_title("Confusion Matrix - Optimized Model (Test Set)")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("True")

    plt.tight_layout()
    plt.savefig(COMBINED_PLOT_OUT, dpi=150)
    print(f"Saved combined comparison grid to {COMBINED_PLOT_OUT}")


if __name__ == "__main__":
    results, classes = evaluate()
    plot_comparison_grid(results, classes)
