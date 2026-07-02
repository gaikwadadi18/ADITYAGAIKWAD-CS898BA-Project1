"""
Homework 2, Part 5: Manual Ground Truth Mask
-------------------------------------------------
Creates a reference ("ground truth") mask of the figure to evaluate
the three segmentation methods against.

The figure's location was identified manually by visual inspection of
the source image (bounding box below). GrabCut is then run, seeded
with that manually-specified box, to refine a pixel-accurate outline
within it -- this is a standard way to hand-annotate a clean
segmentation mask without manually painting every pixel by hand.
"""

import cv2
import numpy as np
import os

os.makedirs("images/segmentation/ground_truth", exist_ok=True)

image = cv2.imread("images/original/HW1_IMG_CS898BA.png")
h, w = image.shape[:2]

# Manually identified bounding box around the figure (x1, y1, x2, y2),
# found by visually inspecting images/original/HW1_IMG_CS898BA.png.
FIGURE_BOX = (400, 440, 580, 1080)
x1, y1, x2, y2 = FIGURE_BOX
rect = (x1, y1, x2 - x1, y2 - y1)

mask = np.zeros((h, w), np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Convert GrabCut's 4-value mask (definite/probable background/foreground)
# into a clean binary mask.
ground_truth = np.where(
    (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
).astype(np.uint8)

cv2.imwrite("images/segmentation/ground_truth/ground_truth_mask.png", ground_truth)

print("Ground truth mask created from manually-identified bounding box "
      f"{FIGURE_BOX}, refined with GrabCut.")
print("Saved: images/segmentation/ground_truth/ground_truth_mask.png")
