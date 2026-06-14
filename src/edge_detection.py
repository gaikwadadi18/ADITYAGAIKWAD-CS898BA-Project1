import cv2
import numpy as np
import os

subset_folder = "images/subsets/subset_1"

output_folder = "images/edge_results"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(subset_folder):
    if file.endswith(".png"):
        image_path = os.path.join(
    subset_folder,
    file
)


image = cv2.imread(image_path)

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

sobelx = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

sobely = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

sobel = cv2.magnitude(
    sobelx,
    sobely
)

cv2.imwrite(
    f"{output_folder}/{file[:-4]}_sobel.png",
    sobel
)

laplacian = cv2.Laplacian(
    gray,
    cv2.CV_64F
)

laplacian = cv2.convertScaleAbs(
    laplacian
)

cv2.imwrite(
    f"{output_folder}/{file[:-4]}_laplacian.png",
    laplacian
)


canny = cv2.Canny(
    gray,
    100,
    200
)


cv2.imwrite(
    f"{output_folder}/{file[:-4]}_canny.png",
    canny
)

kernelx = np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
])

kernely = np.array([
    [1,1,1],
    [0,0,0],
    [-1,-1,-1]
])


prewitt_x = cv2.filter2D(
    gray,
    -1,
    kernelx
)

prewitt_y = cv2.filter2D(
    gray,
    -1,
    kernely
)

prewitt = cv2.addWeighted(
    prewitt_x,
    0.5,
    prewitt_y,
    0.5,
    0
)


cv2.imwrite(
    f"{output_folder}/{file[:-4]}_prewitt.png",
    prewitt
)


print(
    "Edge detection completed."
)

