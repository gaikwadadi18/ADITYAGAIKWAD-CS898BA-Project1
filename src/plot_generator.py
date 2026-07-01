"""
Comparison Plot Generator
--------------------------
For every image in the subset, builds a side-by-side figure showing the
original image next to its Sobel, Laplacian, Canny, and Prewitt outputs,
and saves the figure to images/plots/.

NOTE (fixed after grading feedback): like edge_detection.py, this script
originally only generated ONE comparison plot because the body of the
for-loop was not indented, so it ran once after the loop finished
instead of once per image. That is why only a single plot existed in
images/plots and why the written comparison in read.md was not
actually representative of the full subset.
"""

import cv2
import matplotlib.pyplot as plt
import os

subset_folder = "images/subsets/subset_1"
edge_folder = "images/edge_results"
plot_folder = "images/plots"

os.makedirs(plot_folder, exist_ok=True)

for file in os.listdir(subset_folder):
    if not file.endswith(".png"):
        continue

    base = file[:-4]

    # Load the original image and convert BGR -> RGB for correct
    # display colors in matplotlib.
    original = cv2.imread(os.path.join(subset_folder, file))
    if original is None:
        print(f"Skipping unreadable file: {file}")
        continue
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Load the four edge-detection results generated for this image.
    sobel = cv2.imread(f"{edge_folder}/{base}_sobel.png", 0)
    laplacian = cv2.imread(f"{edge_folder}/{base}_laplacian.png", 0)
    canny = cv2.imread(f"{edge_folder}/{base}_canny.png", 0)
    prewitt = cv2.imread(f"{edge_folder}/{base}_prewitt.png", 0)

    if any(img is None for img in [sobel, laplacian, canny, prewitt]):
        print(f"Skipping {file}: missing edge-detection output")
        continue

    # Build a 1x5 comparison figure: original + four edge detectors.
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 5, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(sobel, cmap="gray")
    plt.title("Sobel")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(laplacian, cmap="gray")
    plt.title("Laplacian")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(canny, cmap="gray")
    plt.title("Canny")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(prewitt, cmap="gray")
    plt.title("Prewitt")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{plot_folder}/{base}_comparison.png")
    plt.close()

print("Comparison plots generated for all images in subset.")
