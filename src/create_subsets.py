import os
import random
import shutil

all_images = []

folders = [
    "images/original",
    "images/converted",
    "images/transformed",
    "images/blurred"
]

for folder in folders:

    for file in os.listdir(folder):

        if file.endswith(".png"):

            all_images.append(
                os.path.join(folder, file)
            )

random.shuffle(all_images)

subset_size = 42

for i in range(4):

    subset_folder = (
        f"images/subsets/subset_{i+1}"
    )

    os.makedirs(
        subset_folder,
        exist_ok=True
    )

    start = i * subset_size
    end = start + subset_size

    subset = all_images[start:end]

    for image in subset:

        shutil.copy(
            image,
            subset_folder
        )

print("4 subsets created.")