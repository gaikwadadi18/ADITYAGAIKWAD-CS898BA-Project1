"""Visualize original vs augmented training images -- for README/report use."""
import matplotlib.pyplot as plt
from PIL import Image
from src.data_pipeline import build_file_list, train_transform, DATA_DIR
import numpy as np

filepaths, labels, classes, class_to_idx = build_file_list(DATA_DIR)

# Pick one sample image per class
sample_paths = {}
for p, l in zip(filepaths, labels):
    c = classes[l]
    if c not in sample_paths:
        sample_paths[c] = p
    if len(sample_paths) == len(classes):
        break

fig, axes = plt.subplots(2, len(classes), figsize=(3 * len(classes), 6))

for i, (c, p) in enumerate(sample_paths.items()):
    img = Image.open(p).convert("RGB")
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"{c}\n(original)", fontsize=10)
    axes[0, i].axis("off")

    aug_tensor = train_transform(img)
    aug_img = (aug_tensor.permute(1, 2, 0).numpy() * 0.5) + 0.5  # undo normalization
    aug_img = np.clip(aug_img, 0, 1)
    axes[1, i].imshow(aug_img)
    axes[1, i].set_title("augmented", fontsize=10)
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("/home/claude/fish_project/outputs/plots/augmentation_examples.png", dpi=150)
print("Saved outputs/plots/augmentation_examples.png")
