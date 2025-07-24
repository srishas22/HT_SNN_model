import numpy as np
import cv2
from typing import Tuple, List

def preprocess_image(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    normalized = blurred.astype(np.float32) / 255.0
    return normalized

def align_images(pui_image: np.ndarray, gm_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(pui_image, None)
    kp2, des2 = sift.detectAndCompute(gm_image, None)

    if des1 is None or des2 is None:
        return pui_image, gm_image

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return pui_image, gm_image

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if homography is not None:
        h, w = gm_image.shape
        aligned_pui = cv2.warpPerspective(pui_image, homography, (w, h))
        return aligned_pui, gm_image

    return pui_image, gm_image

def identify_suspicious_regions(pui_image: np.ndarray, gm_image: np.ndarray, threshold: float = 0.1) -> List[Tuple[int, int, int, int]]:
    if pui_image.shape != gm_image.shape:
        pui_image = cv2.resize(pui_image, (gm_image.shape[1], gm_image.shape[0]))

    diff = cv2.absdiff(pui_image, gm_image)
    _, binary = cv2.threshold(diff, threshold * 255, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    suspicious_regions = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            suspicious_regions.append((x, y, w, h))
    return suspicious_regions

def crop_and_normalize_regions(pui_image: np.ndarray, gm_image: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    pui_crops, gm_crops = [], []
    for x, y, w, h in regions:
        pui_crop = pui_image[y:y+h, x:x+w]
        gm_crop = gm_image[y:y+h, x:x+w]
        pui_resized = cv2.resize(pui_crop, (224, 224))
        gm_resized = cv2.resize(gm_crop, (224, 224))
        pui_crops.append(pui_resized.astype(np.float32) / 255.0)
        gm_crops.append(gm_resized.astype(np.float32) / 255.0)
    return np.array(pui_crops), np.array(gm_crops)
