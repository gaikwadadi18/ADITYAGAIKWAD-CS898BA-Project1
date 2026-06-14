import cv2

image = cv2.imread("images/original/HW1_IMG_CS898BA.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite(
    "images/converted/grayscale.png",
    gray
)

_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

cv2.imwrite(
    "images/converted/binary.png",
    binary
)

hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV
)

cv2.imwrite(
    "images/converted/hsv.png",
    hsv
)

lab = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2LAB
)

cv2.imwrite(
    "images/converted/lab.png",
    lab
)

hls = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HLS
)

cv2.imwrite(
    "images/converted/hls.png",
    hls
)


# Split HSV channels
h, s, v = cv2.split(hsv)

# Equalize V channel
v_eq = cv2.equalizeHist(v)

# Merge channels back
hsv_eq = cv2.merge([h, s, v_eq])

# Convert back to BGR
equalized_rgb = cv2.cvtColor(
    hsv_eq,
    cv2.COLOR_HSV2BGR
)

# Save result
cv2.imwrite(
    "images/converted/equalized_rgb.png",
    equalized_rgb
)

print("All conversions complete.")