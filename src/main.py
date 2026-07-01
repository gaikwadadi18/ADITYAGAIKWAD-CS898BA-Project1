"""
Pipeline Runner
-----------------
Runs the full CS898BA Project 1 pipeline end to end, in order:

1. image_statistics.py  - per-channel pixel statistics -> result/statistics.txt
2. color_conversion.py  - grayscale/binary/HSV/LAB/HLS + histogram equalization
3. affine_transform.py  - rotation/translation/scaling/shearing
4. gaussian_blur.py     - blur every image at 7 sigma values
5. create_subsets.py    - randomly split all generated images into 4 subsets
6. edge_detection.py    - Sobel/Laplacian/Canny/Prewitt on subset_1
7. plot_generator.py    - side-by-side comparison figures for subset_1

Run with: python src/main.py  (from the project root, so relative
paths like "images/..." resolve correctly).
"""

import runpy

STEPS = [
    "src/image_statistics.py",
    "src/color_conversion.py",
    "src/affine_transform.py",
    "src/gaussian_blur.py",
    "src/create_subsets.py",
    "src/edge_detection.py",
    "src/plot_generator.py",
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n=== Running {step} ===")
        runpy.run_path(step, run_name="__main__")
    print("\nPipeline complete.")
