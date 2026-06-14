import cv2
import numpy as np
import os

# Create output folder if it doesn't exist
os.makedirs("images/transformed", exist_ok=True)


# Load Images


images = {
    "original": cv2.imread("images/original/HW1_IMG_CS898BA.png"),
    "grayscale": cv2.imread("images/converted/grayscale.png"),
    "binary": cv2.imread("images/converted/binary.png"),
    "hsv": cv2.imread("images/converted/hsv.png"),
    "lab": cv2.imread("images/converted/lab.png"),
    "hls": cv2.imread("images/converted/hls.png"),
    "equalized": cv2.imread("images/converted/equalized_rgb.png")
}


# Transformation Functions


def rotate_image(image, angle):
    h, w = image.shape[:2]

    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    return cv2.warpAffine(
        image,
        matrix,
        (w, h)
    )


def scale_image(image, scale):
    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale
    )


def translate_image(image, tx, ty):
    h, w = image.shape[:2]

    matrix = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    return cv2.warpAffine(
        image,
        matrix,
        (w, h)
    )


def shear_image(image, shear):
    h, w = image.shape[:2]

    matrix = np.float32([
        [1, shear, 0],
        [0, 1, 0]
    ])

    return cv2.warpAffine(
        image,
        matrix,
        (w, h)
    )


# ORIGINAL


cv2.imwrite(
    "images/transformed/original_rotate15.png",
    rotate_image(images["original"], 15)
)

cv2.imwrite(
    "images/transformed/original_scale12.png",
    scale_image(images["original"], 1.2)
)


# GRAYSCALE


cv2.imwrite(
    "images/transformed/grayscale_translate25.png",
    translate_image(images["grayscale"], 25, 0)
)

cv2.imwrite(
    "images/transformed/grayscale_rotate90.png",
    rotate_image(images["grayscale"], 90)
)


# BINARY


cv2.imwrite(
    "images/transformed/binary_shear015.png",
    shear_image(images["binary"], 0.15)
)

cv2.imwrite(
    "images/transformed/binary_scale08.png",
    scale_image(images["binary"], 0.8)
)


# HSV


cv2.imwrite(
    "images/transformed/hsv_rotate186.png",
    rotate_image(images["hsv"], 186)
)

cv2.imwrite(
    "images/transformed/hsv_translate20.png",
    translate_image(images["hsv"], 0, 20)
)


# LAB


cv2.imwrite(
    "images/transformed/lab_rotate45.png",
    rotate_image(images["lab"], 45)
)

cv2.imwrite(
    "images/transformed/lab_scale14.png",
    scale_image(images["lab"], 1.4)
)


# HLS


cv2.imwrite(
    "images/transformed/hls_shear025.png",
    shear_image(images["hls"], 0.25)
)

cv2.imwrite(
    "images/transformed/hls_rotate75.png",
    rotate_image(images["hls"], 75)
)


# EQUALIZED RGB


cv2.imwrite(
    "images/transformed/equalized_rotate135.png",
    rotate_image(images["equalized"], 135)
)

cv2.imwrite(
    "images/transformed/equalized_scale06.png",
    scale_image(images["equalized"], 0.6)
)

print("14 affine transformations created successfully.")