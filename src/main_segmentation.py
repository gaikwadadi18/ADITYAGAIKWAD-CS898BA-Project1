"""
Homework 2 Pipeline Runner
-----------------------------
Runs the full segmentation pipeline end to end, in order:

1. color_normalization.py    - Part 2: multi-channel color normalization
2. threshold_segmentation.py - Part 3: Otsu + Adaptive thresholding
3. ground_truth_mask.py      - Part 5: manual ground truth mask
4. kmeans_segmentation.py    - Part 4: K-Means clustering (depends on
                                 the ground truth mask for K selection)
5. evaluate_segmentation.py  - Part 5: IoU/Dice + comparison plot

Run with: python src/main_segmentation.py  (from the project root).
"""

import runpy

STEPS = [
    "src/color_normalization.py",
    "src/threshold_segmentation.py",
    "src/ground_truth_mask.py",
    "src/kmeans_segmentation.py",
    "src/evaluate_segmentation.py",
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n=== Running {step} ===")
        runpy.run_path(step, run_name="__main__")
    print("\nSegmentation pipeline complete.")
