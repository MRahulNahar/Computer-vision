import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread('firstlecture\Screenshot 2025-08-15 003515.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image file not found.")

# --- 1. Simple Thresholding ---
# All pixels above 127 become 255 (white), others become 0 (black)
thresh_val = 127
max_val = 255
_, thresh_img = cv2.threshold(img, thresh_val, max_val, cv2.THRESH_BINARY)

# --- 2. Custom Pixel Mapping ---
def pixel_map(val):
    # Example: Map pixels above 200 to 255, below 50 to 0, others to 125
    if val > 200:
        return 255
    elif val < 50:
        return 0
    else:
        return 125

# Vectorize the pixel_map function for performance
pixel_map_vectorized = np.vectorize(pixel_map)
mapped_img = pixel_map_vectorized(img)

# --- 3. Save and Display Results ---
cv2.imwrite('thresholded_output.jpg', thresh_img)
cv2.imwrite('mapped_output.jpg', mapped_img)

# Histogram of original image
plt.subplot(1, 3, 1)
plt.hist(img.ravel(), bins=256, range=(0,255), color='blue')
plt.title('Original Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Count')

# Histogram of mapped image
plt.subplot(1, 3, 2)
plt.hist(mapped_img.ravel(), bins=256, range=(0,255), color='green')
plt.title('Mapped Histogram')
plt.xlabel('Pixel Intensity')

# Histogram of thresholded image
plt.subplot(1, 3, 3)
plt.hist(thresh_img.ravel(), bins=256, range=(0,255), color='red')
plt.title('Thresholded Histogram')
plt.xlabel('Pixel Intensity')

plt.tight_layout()
plt.show()

cv2.imshow('Original', img)
cv2.imshow('Thresholded', thresh_img)
cv2.imshow('Pixel Mapped', mapped_img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
