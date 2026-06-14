# CS898BA Project 1

## Student Information

Name: Aditya Ashokbhai Gaikwad

## Objective

The goal of this project was to perform image analysis and processing using OpenCV. The project involved image statistics, color space conversions, histogram equalization, affine transformations, Gaussian blurring, and edge detection.

## Technologies Used

- Python
- OpenCV
- NumPy
- SciPy
- Matplotlib

## Project Structure

(Insert folder structure)

## Part 2: Image Statistics

Describe the statistics that were computed.

Include results from statistics.txt.

## Color Space Conversions

- Grayscale
- Binary
- HSV
- LAB
- HLS

Discuss each conversion.

## Histogram Equalization

Histogram equalization was applied to the Value (V) channel of the HSV image to normalize illumination and improve contrast. The image was then converted back to RGB/BGR format and saved.

## Affine Transformations

Four affine transformation types were used:

- Rotation
- Translation
- Scaling
- Shearing

A total of 14 unique affine transformations were created.

## Gaussian Blur Analysis

Gaussian blur was applied using sigma values:

0.5
1.0
1.5
2.0
2.5
3.0
3.5

As sigma increased, image smoothing increased while fine detail and edge sharpness decreased.

## Part 3: Edge Detection

### Sobel

Pros:
- Fast
- Detects strong gradients

Cons:
- Thick edges
- Noise sensitive

### Laplacian

Pros:
- Detects edges in all directions

Cons:
- Highly noise sensitive
- Produces double edges

### Canny

Pros:
- Thin edges
- Excellent localization
- Noise reduction built in

Cons:
- Computationally expensive

### Prewitt

Pros:
- Simple
- Fast

Cons:
- Less accurate than Sobel

## Edge Detection Comparison

Canny produced the most useful results for the image set. The figure was clearly outlined while background noise was reduced. Sobel and Prewitt identified major boundaries but produced thicker edge responses. Laplacian generated the largest amount of edge information but also amplified noise.

## Sample Results

Insert six randomly selected comparison plots.

## Conclusion

This project demonstrated several fundamental image analysis techniques including image statistics, color space conversion, contrast enhancement, affine transformations, image smoothing, and edge detection. Among the evaluated edge detectors, Canny provided the best overall balance between edge localization and noise suppression for the given image set.