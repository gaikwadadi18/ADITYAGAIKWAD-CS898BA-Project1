"""
Image Statistics
------------------
Computes basic per-channel pixel statistics (min, max, mean, median,
mode, skew, range, std dev, variance) for the source image and writes
the results to result/statistics.txt so they are reproducible instead
of only being printed to the console and pasted in by hand.
"""

import os
import cv2
import numpy as np
from scipy import stats

image = cv2.imread("images/original/HW1_IMG_CS898BA.png")

# Split into individual color channels (OpenCV loads images as BGR).
b, g, r = cv2.split(image)

channels = {
    "Blue": b,
    "Green": g,
    "Red": r
}

os.makedirs("result", exist_ok=True)

lines = []
for name, channel in channels.items():
    pixels = channel.flatten()

    lines.append(f"\n{name} Channel")
    lines.append(f"Min: {np.min(pixels)}")
    lines.append(f"Max: {np.max(pixels)}")
    lines.append(f"Mean: {np.mean(pixels)}")
    lines.append(f"Median: {np.median(pixels)}")
    lines.append(f"Mode: {stats.mode(pixels, keepdims=True)[0][0]}")
    lines.append(f"Skew: {stats.skew(pixels)}")
    lines.append(f"Range: {np.max(pixels) - np.min(pixels)}")
    lines.append(f"Std Dev: {np.std(pixels)}")
    lines.append(f"Variance: {np.var(pixels)}")

output_text = "\n".join(lines)
print(output_text)

with open("result/statistics.txt", "w") as f:
    f.write(output_text + "\n")

print("\nStatistics saved to result/statistics.txt")
