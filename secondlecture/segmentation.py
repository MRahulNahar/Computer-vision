import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# ---------- 1) Load ----------
path = 'secondlecture\map.png'  # set to the river image
img_bgr = cv2.imread(path)
if img_bgr is None:
    raise FileNotFoundError(path)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
cv2.imwrite('01_grayscale.png', img_gray)

# ---------- 2) Color k-means segmentation (multi-class) ----------
# K determines number of color clusters; tune 3–6
Z = img_bgr.reshape((-1,3)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
K = 3
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
seg = centers[labels.flatten()].reshape(img_bgr.shape).astype(np.uint8)
cv2.imwrite('02_kmeans_seg.png', seg)  # color-coded map
# If water is typically darker, water cluster will be one of the darker centers.
# You can pick the darkest cluster as "water mask":
water_cluster = np.argmin(np.mean(centers, axis=1))  # darkest RGB mean
mask_kmeans = (labels.flatten()==water_cluster).reshape(img_gray.shape).astype(np.uint8)*255
cv2.imwrite('02b_kmeans_water_mask.png', mask_kmeans)

# ---------- 3) Binary water mask via thresholding ----------
# Denoise and apply both Otsu and Adaptive; choose the better result
blur = cv2.GaussianBlur(img_gray, (5,5), 0)

# Otsu (global, auto)
_, mask_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# If water appears dark, use THRESH_BINARY_INV; flip to THRESH_BINARY if water is bright. Adjust as needed

# Adaptive (handles uneven illumination)
mask_adapt = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
    blockSize=51, C=2
)

# ---------- 4) Morphology to clean and connect channels ----------
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask_clean = cv2.morphologyEx(mask_adapt, cv2.MORPH_OPEN, kernel, iterations=1)  # remove speckle
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2) # connect gaps
cv2.imwrite('03a_mask_otsu.png', mask_otsu)
cv2.imwrite('03b_mask_adaptive.png', mask_adapt)
cv2.imwrite('03c_mask_clean.png', mask_clean)

# ---------- 5) Optional watershed to fill breaks ----------
# Use distance transform + markers to ensure continuous channels
dist = cv2.distanceTransform(mask_clean, cv2.DIST_L2, 5)          # expects 8-bit single-channel 0/255
peaks = peak_local_max(dist, min_distance=7, labels=mask_clean)   # tune min_distance for seed density
markers = np.zeros(dist.shape, dtype=np.int32)
markers[tuple(peaks.T)] = np.arange(1, len(peaks)+1)
labels_ws = watershed(-dist, markers, mask=mask_clean.astype(bool))
# Convert labels to a binary water mask (anything labeled is water)
mask_ws = (labels_ws > 0).astype(np.uint8)*255
cv2.imwrite('04_watershed_mask.png', mask_ws)

# ---------- 6) Final export ----------
# Optionally combine k-means water and threshold mask to be conservative
mask_combined = cv2.bitwise_and(mask_ws, mask_kmeans)
cv2.imwrite('05_final_water_mask.png', mask_combined)
