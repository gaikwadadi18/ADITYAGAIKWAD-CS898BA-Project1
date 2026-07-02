"""
Homework 2, Part 3: Threshold-Based Segmentation
----------------------------------------------------
Applies two classical thresholding techniques to the grayscale
version of the normalized color image from Part 2:

1. Otsu's global thresholding - automatically picks a single
   intensity threshold that best separates foreground from background
   across the WHOLE image.
2. Adaptive Gaussian thresholding - computes a locally-varying
   threshold for each pixel neighborhood, which should handle uneven
   illumination (e.g. shadows on one side of the yard) better than a
   single global threshold.

For each method, saves the binary mask and the segmented foreground
(original-color pixels kept where the mask is foreground, black
elsewhere).
"""

import cv2
import os

os.makedirs("images/segmentation/otsu", exist_ok=True)
os.makedirs("images/segmentation/adaptive", exist_ok=True)

normalized = cv2.imread("images/segmentation/normalized/normalized_color.png")
gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)

# --- Otsu's global thresholding ---
otsu_value, otsu_mask = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
cv2.imwrite("images/segmentation/otsu/otsu_mask.png", otsu_mask)

otsu_foreground = cv2.bitwise_and(normalized, normalized, mask=otsu_mask)
cv2.imwrite("images/segmentation/otsu/otsu_foreground.png", otsu_foreground)

print(f"Otsu threshold value chosen automatically: {otsu_value}")

# --- Adaptive Gaussian thresholding ---
adaptive_mask = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=25,
    C=5,
)
cv2.imwrite("images/segmentation/adaptive/adaptive_mask.png", adaptive_mask)

adaptive_foreground = cv2.bitwise_and(normalized, normalized, mask=adaptive_mask)
cv2.imwrite("images/segmentation/adaptive/adaptive_foreground.png", adaptive_foreground)

print("Threshold-based segmentation complete.")
print("Saved Otsu and Adaptive masks + foreground extractions.")
