import cv2
import os
from pathlib import Path

os.makedirs(
    "images/blurred",
    exist_ok=True
)

sigma_values = [
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5
]

image_files = []

folders = [
    "images/original",
    "images/converted",
    "images/transformed"
]

for folder in folders:

    for file in os.listdir(folder):

        if file.endswith(".png"):

            image_files.append(
                os.path.join(folder, file)
            )

print(
    "Images Found:",
    len(image_files)
)

for image_path in image_files:

    image = cv2.imread(image_path)

    filename = Path(
        image_path
    ).stem

    for sigma in sigma_values:

        blurred = cv2.GaussianBlur(
            image,
            (0, 0),
            sigmaX=sigma
        )

        output_name = (
            f"{filename}_sigma_{sigma}.png"
        )

        cv2.imwrite(
            f"images/blurred/{output_name}",
            blurred
        )

print(
    "147 blurred images created successfully."
)