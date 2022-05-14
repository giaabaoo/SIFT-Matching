import cv2
import numpy as np
import pdb
from pathlib import Path
# if you can not import library, type the following command
# pip install opencv-python numpy


# Load Image
image_path = "./Code/objets/objet1.jpg"
image = cv2.imread(image_path)

# Convert colored image to grayscale image
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Normalize image range to [0..1]
grayscale = grayscale.astype(np.float64) / 255.0

# Compute Image Gradient w.r.t x and y axis
Image_x = cv2.Sobel(grayscale, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3) # Your Code Here
Image_y = cv2.Sobel(grayscale, ddepth=cv2.CV_16S, dx=0, dy=1, ksize=3) # Your Code Here

# pdb.set_trace()

# Compute Second Derivative
Image_xx = Image_x * Image_x
Image_xy = Image_x * Image_y
Image_yy = Image_y * Image_y

# Apply Gaussian Kernel
window_size = (5, 5)
sigma = 500
Image_xx = cv2.GaussianBlur(Image_xx, window_size, sigma)
Image_xy = cv2.GaussianBlur(Image_xy, window_size, sigma)
Image_yy = cv2.GaussianBlur(Image_yy, window_size, sigma)

# Compute Harris Response
k = 0.05
R = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.float64)
offset = window_size[0]//2
keypoint = image.copy()
threshold = 100.0 # Threshold to determined a corner

count = 0

for i in range(offset, image.shape[0]-offset):
    for j in range(offset, image.shape[1]-offset):
        # Get Ixx, Ixy, and Iyy
        Ixx = Image_xx[i-offset:i+offset, j-offset:j+offset]
        Ixy = Image_xy[i-offset:i+offset, j-offset:j+offset]
        Iyy = Image_yy[i-offset:i+offset, j-offset:j+offset]

        # Compute Sum of Ixx, Ixy, Iyy
        Sxx = Ixx.sum()
        Sxy = Ixy.sum()
        Syy = Iyy.sum()

        # # Compute Matrix H
        # H = ... # Your Code Here

        # Compute det and trace of H
        det = (Sxx * Syy) - (Sxy**2) # Your Code Here
        trace = Sxx + Syy # Your Code Here

        R[i][j] = det - k * (trace ** 2)
        if R[i][j] > threshold: # Found a corner
            cv2.circle(keypoint, (j,i), 1, (0,255,0))
            count += 1

# cv2.imshow("Heatmap", R)
# cv2.imshow("Keypoint", keypoint)
# cv2.waitKey(0)
print("Number of detected corners:", count)
Path("./Q1").mkdir(parents=True, exist_ok=True)
cv2.imwrite("./Q1/q1_2_heatmap.jpg", R)
cv2.imwrite("./Q1/q1_2_corners.jpg", keypoint)

