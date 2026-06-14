import cv2
import numpy as np
from scipy import stats

image = cv2.imread("images/original/HW1_IMG_CS898BA.png")

b,g,r = cv2.split(image)

channels = {
    "Blue": b,
    "Green": g,
    "Red": r
}

for name, channel in channels.items():

    pixels = channel.flatten()

    print(f"\n{name} Channel")

    print("Min:", np.min(pixels))
    print("Max:", np.max(pixels))
    print("Mean:", np.mean(pixels))
    print("Median:", np.median(pixels))
    print("Mode:", stats.mode(pixels, keepdims=True)[0][0])
    print("Skew:", stats.skew(pixels))
    print("Range:", np.max(pixels)-np.min(pixels))
    print("Std Dev:", np.std(pixels))
    print("Variance:", np.var(pixels))