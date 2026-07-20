"""
Part 2: Data Preprocessing & Augmentation
CS 898BA - Homework Three - Fish Classification

- Loads the fish image dataset from a class-per-folder directory structure
- Creates a stratified 70/15/15 train/val/test split
- Resizes images to IMG_SIZE x IMG_SIZE and normalizes to [-1, 1]
- Applies data augmentation (random flips, rotations, brightness/contrast jitter)
  to the TRAINING set only
- Computes class weights to handle class imbalance (used later in the loss fn)
"""

import os
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/home/claude/fish_data/Fish"   # class-per-folder root
IMG_SIZE = 96                               # 96x96, chosen for fast CPU training
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0  # single-core sandbox; bump this up on your own machine

random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Build file list + stratified split
# ---------------------------------------------------------------------------
def build_file_list(data_dir):
    classes = sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    filepaths, labels = [], []
    for c in classes:
        for f in sorted(Path(data_dir, c).iterdir()):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                filepaths.append(str(f))
                labels.append(class_to_idx[c])

    return filepaths, labels, classes, class_to_idx


def stratified_split(filepaths, labels, train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=SEED):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    # First split off the test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        filepaths, labels, test_size=test_frac, stratify=labels, random_state=seed
    )
    # Then split train_val into train / val
    val_relative = val_frac / (train_frac + val_frac)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_relative,
        stratify=train_val_labels, random_state=seed
    )

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
# Normalize to [-1, 1] (ToTensor already gives [0,1], so mean=0.5/std=0.5 maps it to [-1,1])
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FishDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


def get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    filepaths, labels, classes, class_to_idx = build_file_list(data_dir)
    (train_p, train_l), (val_p, val_l), (test_p, test_l) = stratified_split(filepaths, labels)

    train_ds = FishDataset(train_p, train_l, transform=train_transform)
    val_ds = FishDataset(val_p, val_l, transform=eval_transform)
    test_ds = FishDataset(test_p, test_l, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Class weights (inverse frequency) to counter imbalance -- used in the loss fn later
    counts = [train_l.count(i) for i in range(len(classes))]
    total = sum(counts)
    class_weights = torch.tensor([total / (len(classes) * c) for c in counts], dtype=torch.float32)

    info = {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "counts_total": {c: labels.count(i) for c, i in class_to_idx.items()},
        "counts_train": {c: train_l.count(i) for c, i in class_to_idx.items()},
        "counts_val": {c: val_l.count(i) for c, i in class_to_idx.items()},
        "counts_test": {c: test_l.count(i) for c, i in class_to_idx.items()},
        "class_weights": class_weights.tolist(),
        "img_size": IMG_SIZE,
    }

    return train_loader, val_loader, test_loader, class_weights, info


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_weights, info = get_dataloaders()

    print("Classes:", info["classes"])
    print("Total images per class:", info["counts_total"])
    print("Train split:", info["counts_train"])
    print("Val split:  ", info["counts_val"])
    print("Test split: ", info["counts_test"])
    print("Class weights (inverse frequency):", [round(w, 3) for w in info["class_weights"]])

    xb, yb = next(iter(train_loader))
    print("Batch shape:", xb.shape, "labels:", yb.shape)
    print("Pixel value range after normalization: [%.3f, %.3f]" % (xb.min().item(), xb.max().item()))

    os.makedirs("/home/claude/fish_project/outputs/reports", exist_ok=True)
    with open("/home/claude/fish_project/outputs/reports/split_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("\nSaved split info to outputs/reports/split_info.json")
