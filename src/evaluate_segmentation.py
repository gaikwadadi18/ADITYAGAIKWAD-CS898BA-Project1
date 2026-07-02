"""
Homework 2, Part 5: Quantitative Evaluation + Comparison Plot
-------------------------------------------------------------
Computes Intersection over Union (IoU / Jaccard Index) and the Dice
Coefficient for each of the 3 segmentation methods (Otsu, Adaptive,
K-Means) against the manual ground truth mask, prints/saves the
results, and builds a single comparison figure showing the original
image, the normalized image, and the 3 segmentation masks side by
side.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("result", exist_ok=True)
os.makedirs("images/segmentation/comparison", exist_ok=True)


def iou_score(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union > 0 else 0.0


def dice_score(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return (2 * intersection) / denom if denom > 0 else 0.0


ground_truth = cv2.imread(
    "images/segmentation/ground_truth/ground_truth_mask.png", 0
)

masks = {
    "Otsu": cv2.imread("images/segmentation/otsu/otsu_mask.png", 0),
    "Adaptive": cv2.imread("images/segmentation/adaptive/adaptive_mask.png", 0),
    "K-Means": cv2.imread("images/segmentation/kmeans/kmeans_mask.png", 0),
}

lines = ["Segmentation Evaluation vs Manual Ground Truth Mask", ""]
lines.append(f"{'Method':<10} {'IoU':>8} {'Dice':>8}")
for name, mask in masks.items():
    iou = iou_score(mask, ground_truth)
    dice = dice_score(mask, ground_truth)
    line = f"{name:<10} {iou:>8.4f} {dice:>8.4f}"
    print(line)
    lines.append(line)

with open("result/segmentation_metrics.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print("\nSaved: result/segmentation_metrics.txt")

# --- Comparison plot: original, normalized, Otsu, Adaptive, K-Means ---
original = cv2.cvtColor(
    cv2.imread("images/original/HW1_IMG_CS898BA.png"), cv2.COLOR_BGR2RGB
)
normalized = cv2.cvtColor(
    cv2.imread("images/segmentation/normalized/normalized_color.png"),
    cv2.COLOR_BGR2RGB,
)

panels = [
    ("Original", original, False),
    ("Normalized (Part 2)", normalized, False),
    ("Otsu Mask", masks["Otsu"], True),
    ("Adaptive Mask", masks["Adaptive"], True),
    ("K-Means Mask", masks["K-Means"], True),
]

plt.figure(figsize=(24, 5))
for i, (title, img, is_gray) in enumerate(panels, start=1):
    plt.subplot(1, len(panels), i)
    if is_gray:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.savefig("images/segmentation/comparison/segmentation_comparison.png")
plt.close()

print("Saved: images/segmentation/comparison/segmentation_comparison.png")
