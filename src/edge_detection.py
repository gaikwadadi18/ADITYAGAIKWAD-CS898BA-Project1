"""
Edge Detection Pipeline
------------------------
Applies four edge detectors (Sobel, Laplacian, Canny, Prewitt) to every
image in a subset folder and saves each result to images/edge_results/.

NOTE (fixed after grading feedback): the original version of this script
only processed the LAST file returned by os.listdir() because the
processing code was not indented inside the for-loop. That bug meant
only 1 of 42 images in the subset was ever analyzed, which is why the
edge-detection comparison and the written analysis in read.md were
based on a single (very dark, heavily blurred) test image instead of
a representative sample.
"""

import cv2
import numpy as np
import os

subset_folder = "images/subsets/subset_1"
output_folder = "images/edge_results"

os.makedirs(output_folder, exist_ok=True)

# Loop over every PNG in the subset and run all four detectors on it.
for file in os.listdir(subset_folder):
    if not file.endswith(".png"):
        continue

    image_path = os.path.join(subset_folder, file)
    base_name = file[:-4]

    # Load the image and convert to grayscale since all four detectors
    # operate on single-channel intensity data.
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping unreadable file: {file}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Sobel: first-derivative gradient in x and y, combined into
    # a gradient magnitude image. Good at highlighting strong,
    # directional edges but produces relatively thick edge responses.
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.convertScaleAbs(sobel)
    cv2.imwrite(f"{output_folder}/{base_name}_sobel.png", sobel)

    # --- Laplacian: second-derivative operator, sensitive to edges in
    # all directions at once, but also very sensitive to noise.
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    cv2.imwrite(f"{output_folder}/{base_name}_laplacian.png", laplacian)

    # --- Canny: multi-stage detector (smoothing, gradient, non-max
    # suppression, hysteresis thresholding). Thresholds are fixed at
    # 100/200 here; on very dark or low-contrast images this can
    # suppress almost all edges, which is exactly what we see in the
    # analysis below.
    canny = cv2.Canny(gray, 100, 200)
    cv2.imwrite(f"{output_folder}/{base_name}_canny.png", canny)

    # --- Prewitt: similar to Sobel but with a simpler, unweighted
    # 3x3 kernel, so it is faster but slightly less accurate.
    kernelx = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])
    kernely = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ])
    prewitt_x = cv2.filter2D(gray, -1, kernelx)
    prewitt_y = cv2.filter2D(gray, -1, kernely)
    prewitt = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
    cv2.imwrite(f"{output_folder}/{base_name}_prewitt.png", prewitt)

print("Edge detection completed for all images in subset.")
