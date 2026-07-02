"""
Homework 2, Part 2: Multi-Channel Color Normalization
--------------------------------------------------------
Loads the original Homework 1 image, splits it into its three raw
color channels (B, G, R), applies histogram equalization to EACH
channel independently (not just one channel of HSV/LAB as in
Homework 1), and merges them back into a fully normalized color
image. This maximizes contrast across the entire color spectrum
rather than only along a single brightness axis, which matters for
the doorbell camera's harsh, uneven outdoor lighting.

This normalized image becomes the input for every segmentation
method in Part 3 and Part 4.
"""

import cv2
import os

os.makedirs("images/segmentation/normalized", exist_ok=True)

image = cv2.imread("images/original/HW1_IMG_CS898BA.png")

# Split into the three raw color channels.
b, g, r = cv2.split(image)

# Equalize each channel independently. Unlike Homework 1 (which only
# equalized the HSV Value channel), this stretches contrast in every
# color channel separately, which can shift color balance slightly
# but maximizes contrast across the whole image.
b_eq = cv2.equalizeHist(b)
g_eq = cv2.equalizeHist(g)
r_eq = cv2.equalizeHist(r)

# Merge the equalized channels back into a color image.
normalized = cv2.merge([b_eq, g_eq, r_eq])

cv2.imwrite("images/segmentation/normalized/normalized_color.png", normalized)

print("Multi-channel color normalization complete.")
print("Saved: images/segmentation/normalized/normalized_color.png")
