import cv2
import numpy as np

# --- 1. Load Image ---
# Use a grayscale image for simplicity
img = cv2.imread('firstlecture\Screenshot 2025-08-15 003515.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image file not found.")

# --- 2. Pixel Mapping Function ---
def pixel_map(val):
    # Example mapping: set high to 200, low to 50, others to 120
    if val > 180:
        return 255
    elif val < 80:
        return 0
    else:
        return 125

pixel_map_vectorized = np.vectorize(pixel_map)
mapped_img = pixel_map_vectorized(img)

# --- 3. Thresholding on Mapped Image ---
# E.g., All pixels above 100 become 255, others become 0
thresh_val = 125
max_val = 255
_, thresh_img = cv2.threshold(mapped_img.astype(np.uint8), thresh_val, max_val, cv2.THRESH_BINARY)

# --- 4. Save and Display ---
cv2.imwrite('pixel_mapped.jpg', mapped_img.astype(np.uint8))
cv2.imwrite('pixel_mapped_thresholded.jpg', thresh_img)

cv2.imshow('Mapped Image', mapped_img.astype(np.uint8))
cv2.imshow('Thresholded Output', thresh_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

