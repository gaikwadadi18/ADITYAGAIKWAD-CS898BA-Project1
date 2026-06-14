import cv2
import matplotlib.pyplot as plt
import os

subset_folder = "images/subsets/subset_1"
edge_folder = "images/edge_results"
plot_folder = "images/plots"

os.makedirs(plot_folder, exist_ok=True)

for file in os.listdir(subset_folder):

    if file.endswith(".png"):

        original = cv2.imread(
    os.path.join(subset_folder, file)
)

original = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)

base = file[:-4]
sobel = cv2.imread(
    f"{edge_folder}/{base}_sobel.png",
    0
)

laplacian = cv2.imread(
    f"{edge_folder}/{base}_laplacian.png",
    0
)

canny = cv2.imread(
    f"{edge_folder}/{base}_canny.png",
    0
)

prewitt = cv2.imread(
    f"{edge_folder}/{base}_prewitt.png",
    0
)

plt.figure(figsize=(20,5))
plt.subplot(1,5,1)
plt.imshow(original)
plt.title("Original")
plt.axis("off")
plt.subplot(1,5,2)
plt.imshow(
    sobel,
    cmap="gray"
)
plt.title("Sobel")
plt.axis("off")

plt.subplot(1,5,3)

plt.imshow(
    laplacian,
    cmap="gray"
)

plt.title("Laplacian")

plt.axis("off")

plt.subplot(1,5,4)

plt.imshow(
    canny,
    cmap="gray"
)

plt.title("Canny")

plt.axis("off")

plt.subplot(1,5,5)

plt.imshow(
    prewitt,
    cmap="gray"
)

plt.title("Prewitt")

plt.axis("off")


plt.tight_layout()

plt.savefig(
    f"{plot_folder}/{base}_comparison.png"
)

plt.close()




