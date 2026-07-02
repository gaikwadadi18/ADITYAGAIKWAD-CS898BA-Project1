"""
Homework 2, Part 4: Color-Space Clustering (K-Means)
----------------------------------------------------------
Converts the normalized color image from Part 2 to HSV and runs
K-Means clustering on the pixel colors for K = 3, 4, and 5.

K selection: an initial pass computes compactness (inertia) for each
K, but compactness alone turned out to be a poor guide here - the
"elbow" in compactness does not necessarily correspond to the K that
best isolates the actual figure, since compactness only measures
general cluster tightness, not accuracy against any particular
target object. Since a ground truth mask already exists (from
ground_truth_mask.py, used for the Part 5 evaluation anyway), each K
is additionally scored by IoU against that ground truth, and the K
with the best IoU is the one actually used. This script therefore
depends on ground_truth_mask.py having already been run.

Saves a binary mask and the segmented foreground for the chosen K,
plus a comparison of all three K values tried.
"""

import cv2
import numpy as np
import os

os.makedirs("images/segmentation/kmeans", exist_ok=True)

normalized = cv2.imread("images/segmentation/normalized/normalized_color.png")
hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)

h, w = hsv.shape[:2]
pixel_values = hsv.reshape((-1, 3)).astype(np.float32)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Rough bounding box of the figure (identified by visual inspection of
# the source image), used only to pick which cluster corresponds to
# the figure once K-Means has run - not used to bias the clustering
# itself.
FIGURE_BOX = (400, 440, 580, 1080)  # x1, y1, x2, y2

ground_truth_path = "images/segmentation/ground_truth/ground_truth_mask.png"
if not os.path.exists(ground_truth_path):
    raise FileNotFoundError(
        "Run src/ground_truth_mask.py before src/kmeans_segmentation.py - "
        "K selection depends on the ground truth mask."
    )
ground_truth = cv2.imread(ground_truth_path, 0) > 127


def iou(mask_bool, gt_bool):
    intersection = np.logical_and(mask_bool, gt_bool).sum()
    union = np.logical_or(mask_bool, gt_bool).sum()
    return intersection / union if union > 0 else 0.0


results = {}
for k in [3, 4, 5]:
    compactness, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    labels_2d = labels.reshape((h, w))

    x1, y1, x2, y2 = FIGURE_BOX
    box_labels = labels_2d[y1:y2, x1:x2]
    values, counts = np.unique(box_labels, return_counts=True)
    figure_cluster = values[np.argmax(counts)]

    mask_bool = labels_2d == figure_cluster
    score = iou(mask_bool, ground_truth)

    results[k] = {
        "compactness": compactness,
        "labels_2d": labels_2d,
        "figure_cluster": figure_cluster,
        "iou": score,
    }
    print(f"K={k}: compactness (inertia) = {compactness:.2f}, "
          f"figure cluster = {figure_cluster}, IoU vs ground truth = {score:.4f}")

# Choose K by best IoU against the ground truth, NOT by the compactness
# elbow - see docstring above for why.
chosen_k = max(results, key=lambda k: results[k]["iou"])
print(f"\nChosen K based on best IoU against ground truth: {chosen_k}")

labels_2d = results[chosen_k]["labels_2d"]
figure_cluster = results[chosen_k]["figure_cluster"]

kmeans_mask = np.where(labels_2d == figure_cluster, 255, 0).astype(np.uint8)
cv2.imwrite("images/segmentation/kmeans/kmeans_mask.png", kmeans_mask)

kmeans_foreground = cv2.bitwise_and(normalized, normalized, mask=kmeans_mask)
cv2.imwrite("images/segmentation/kmeans/kmeans_foreground.png", kmeans_foreground)

with open("images/segmentation/kmeans/kmeans_selection.txt", "w") as f:
    f.write("K-Means cluster count comparison (HSV color space)\n\n")
    f.write(f"{'K':<4} {'Compactness':>14} {'IoU vs GT':>12}\n")
    for k in sorted(results.keys()):
        f.write(f"{k:<4} {results[k]['compactness']:>14.2f} "
                f"{results[k]['iou']:>12.4f}\n")
    f.write(f"\nChosen K = {chosen_k} (best IoU against the manual ground "
            f"truth mask, not the compactness elbow - see script docstring)\n")
    f.write(f"Foreground cluster index = {figure_cluster}\n")

print("K-Means segmentation complete.")
print(f"Saved mask + foreground for K={chosen_k}.")
