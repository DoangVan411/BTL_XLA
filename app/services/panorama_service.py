
"""
Panorama stitching service.

Why:
- Encapsulate the panorama pipeline behind a clean, testable interface (SRP).
- Depend on abstractions (DIP): accept algorithmic components via composition.
"""

from typing import List, Tuple

import cv2
import numpy as np

from app.sift import SIFT
from app.matcher import knn_match, Match
from app.homography import find_homography_ransac
from app.transform import perspective_transform, warp_perspective


class PanoramaService:
    """
    Orchestrates the panorama pipeline:
    SIFT -> 2-NN Matching -> Lowe's Ratio -> RANSAC(H) -> Warping -> Blending -> Cropping.
    """

    def __init__(self, sift: SIFT):
        # Rationale: inject SIFT to allow swapping implementations in tests/benchmarks.
        self.sift = sift

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect keypoints and compute descriptors."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.sift.detectAndCompute(gray, None)

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[Match]:
        """
        Perform 2-NN matching and Lowe's ratio test.

        Why:
        - Ratio test removes ambiguous matches, improving RANSAC stability.
        """
        matches = knn_match(desc1, desc2, k=2)
        good: List[Match] = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.7 * n.distance:
                    good.append(m)
        return good

    def find_homography_ransac(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], matches: List[Match]):
        """
        Estimate homography with custom RANSAC + DLT.

        Why:
        - Robust to outliers, critical for correct warping.
        """
        if len(matches) < 4:
            return None, None
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = find_homography_ransac(src_pts, dst_pts, ransac_reproj_threshold=5.0, max_iters=2000)
        return H, mask

    def warp_and_blend(self, img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Warp img2 onto a common canvas and alpha-blend with img1.

        Why:
        - Simple gradient blending hides seams for a more natural panorama.
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

        corners2_transformed = perspective_transform(corners2, H)
        all_corners = np.concatenate((corners1, corners2_transformed), axis=0)

        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        output_shape = (x_max - x_min, y_max - y_min)

        panorama = warp_perspective(img2, translation.dot(H), output_shape)

        y_start = max(0, -y_min)
        y_end = min(panorama.shape[0], h1 - y_min)
        x_start = max(0, -x_min)
        x_end = min(panorama.shape[1], w1 - x_min)

        img1_y_start = max(0, -(-y_min))
        img1_x_start = max(0, -(-x_min))
        img1_region = img1[img1_y_start:img1_y_start + (y_end - y_start), img1_x_start:img1_x_start + (x_end - x_start)]
        pano_region = panorama[y_start:y_end, x_start:x_end]

        gray_pano = cv2.cvtColor(pano_region, cv2.COLOR_BGR2GRAY)
        mask2_region = (gray_pano > 1).astype(np.uint8)
        mask1_region = np.ones(img1_region.shape[:2], dtype=np.uint8)
        overlap_mask = cv2.bitwise_and(mask1_region, mask2_region)

        if np.sum(overlap_mask) > 0:
            overlap_cols = np.where(np.sum(overlap_mask, axis=0) > 0)[0]
            if len(overlap_cols) > 0:
                col_start = overlap_cols[0]
                col_end = overlap_cols[-1]
                blend_width = col_end - col_start + 1
                for col in range(col_start, col_end + 1):
                    alpha = (col - col_start) / max(1, blend_width)
                    for row in range(overlap_mask.shape[0]):
                        if overlap_mask[row, col]:
                            pano_region[row, col] = (
                                (1 - alpha) * img1_region[row, col].astype(float) +
                                alpha * pano_region[row, col].astype(float)
                            ).astype(np.uint8)

        non_overlap_mask = (mask2_region == 0)
        pano_region[non_overlap_mask] = img1_region[non_overlap_mask]
        panorama[y_start:y_end, x_start:x_end] = pano_region
        return panorama

    @staticmethod
    def crop_black_borders(img: np.ndarray) -> np.ndarray:
        """Crop black borders introduced by perspective warp."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return img
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        padding = 2
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        return img[y:y + h, x:x + w]

    def stitch(self, images: List[np.ndarray]) -> np.ndarray:
        """Stitch a list of images into a panorama."""
        if len(images) < 2:
            raise ValueError("At least two images are required for stitching.")

        result = images[0]
        for i in range(1, len(images)):
            kp1, desc1 = self.detect_and_compute(result)
            kp2, desc2 = self.detect_and_compute(images[i])
            matches = self.match_features(desc1, desc2)
            if len(matches) < 10:
                # Avoid poor homography estimation when matches are too few.
                continue
            H, mask = self.find_homography_ransac(kp1, kp2, matches)
            if H is None:
                continue
            result = self.warp_and_blend(result, images[i], H)
        return self.crop_black_borders(result)


