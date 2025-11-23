import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


class SIFT:
    """
    Triển khai SIFT tối giản (không dùng cv2.SIFT) với các bước chính:
    1) Gaussian/DoG Pyramid
    2) Phát hiện điểm cực trị 3D trong không gian (x, y, scale)
    3) Lọc điểm theo độ tương phản và phản hồi biên (edge response)
    4) Gán hướng chính (orientation)
    5) Tính descriptor 128 chiều (4x4 ô, mỗi ô 8 bins)

    Lưu ý:
    - Đây là bản tối giản, hướng tới tính đúng đắn/đào tạo hơn là tối ưu hiệu năng.
    - Có sử dụng một số primitive từ OpenCV (GaussianBlur, resize) nhưng KHÔNG dùng SIFT có sẵn.
    """

    def __init__(
        self,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        sigma: float = 1.6,
        max_octaves: Optional[int] = None,
    ):
        # Tham số giống với SIFT của OpenCV để dễ thay thế
        self.n_octave_layers = int(n_octave_layers)
        self.contrast_threshold = float(contrast_threshold)
        self.edge_threshold = float(edge_threshold)
        self.sigma = float(sigma)
        self.max_octaves = max_octaves  # Nếu None sẽ suy ra theo kích thước ảnh

        # Số ảnh Gaussian trong mỗi octave = n_octave_layers + 3 (chuẩn SIFT)
        self.num_scales = self.n_octave_layers + 3
        self.k = 2 ** (1.0 / self.n_octave_layers)  # hệ số scale giữa các level

    # -------------------------- PUBLIC API -------------------------- #
    def detectAndCompute(
        self, image_gray: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        - image_gray: ảnh xám uint8 (H, W)
        - mask: (không dùng trong bản tối giản)
        Trả về:
        - keypoints: danh sách cv2.KeyPoint
        - descriptors: ndarray (N, 128) float32 hoặc None nếu không có kp
        """
        assert (
            image_gray.ndim == 2
        ), "detectAndCompute yêu cầu ảnh xám (H, W). Hãy chuyển trước khi gọi."

        # Chuẩn hoá ảnh về float32 [0, 1]
        img = image_gray.astype(np.float32) / 255.0

        # Suy ra số octaves theo kích thước ảnh (tối thiểu 1, tối đa theo max_octaves nếu có)
        min_hw = min(img.shape[:2])
        est_octaves = max(1, int(np.floor(np.log2(min_hw))) - 3)  # trừ bớt để tránh quá nhiều
        num_octaves = self.max_octaves if self.max_octaves is not None else max(1, min(est_octaves, 8))

        # 1) Gaussian Pyramid -> tạo các octaves, mỗi octave có num_scales ảnh Gaussian
        gauss_pyr, sigmas_per_level = self._build_gaussian_pyramid(img, num_octaves)

        # 2) DoG Pyramid
        dog_pyr = self._build_dog_pyramid(gauss_pyr)

        # 3) Phát hiện điểm cực trị
        raw_keypoints = self._find_scale_space_extrema(dog_pyr, sigmas_per_level)

        if not raw_keypoints:
            return [], None

        # 4) Gán hướng 
        oriented_keypoints = self._assign_orientations(gauss_pyr, raw_keypoints)

        if not oriented_keypoints:
            return [], None

        # 5) Descriptor 128 chiều
        descriptors = self._compute_descriptors(gauss_pyr, oriented_keypoints)

        return oriented_keypoints, descriptors

    # ----------------------- GAUSSIAN/DoG PYRAMID ----------------------- #
    def _build_gaussian_pyramid(
        self, base_img: np.ndarray, num_octaves: int
    ) -> Tuple[List[List[np.ndarray]], List[List[float]]]:
        """
        Xây Gaussian pyramid:
        - Mỗi octave: num_scales ảnh Gaussian
        - sigma tăng theo k^s, dùng tích luỹ sigma_diff cho mỗi level
        Trả về: (pyramid, sigmas_per_level)
          - pyramid: list[octave][scale] -> ảnh float32
          - sigmas_per_level: list[octave][scale] -> sigma tuyệt đối tại level
        """
        pyr: List[List[np.ndarray]] = []
        sigmas: List[List[float]] = []

        # sigma cơ sở cho level 0 trong mỗi octave
        for o in range(num_octaves):
            octave_imgs: List[np.ndarray] = []
            octave_sigmas: List[float] = []

            if o == 0:
                base = base_img.copy()
            else:
                # xuống mẫu bằng 2 từ ảnh level (o-1, s = n_octave_layers)
                base_prev_oct_last = pyr[o - 1][self.n_octave_layers]
                h, w = base_prev_oct_last.shape
                base = cv2.resize(base_prev_oct_last, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)

            # Duy trì sigma tuyệt đối cho mỗi scale trong octave
            sigma_prev = 0.0
            for s in range(self.num_scales):
                sigma_total = self.sigma * (self.k ** s)
                sigma_diff = math.sqrt(max(sigma_total**2 - sigma_prev**2, 1e-9))
                img_blur = cv2.GaussianBlur(base if s == 0 else octave_imgs[-1], (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)
                octave_imgs.append(img_blur)
                octave_sigmas.append(sigma_total)
                sigma_prev = sigma_total

            pyr.append(octave_imgs)
            sigmas.append(octave_sigmas)

        return pyr, sigmas

    def _build_dog_pyramid(self, gauss_pyr: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """
        Tạo DoG = G(s+1) - G(s) cho mỗi octave.
        """
        dog_pyr: List[List[np.ndarray]] = []
        for octave_imgs in gauss_pyr:
            dogs = []
            for s in range(1, len(octave_imgs)):
                dogs.append(octave_imgs[s] - octave_imgs[s - 1])
            dog_pyr.append(dogs)
        return dog_pyr

    # ----------------------- KEYPOINT DETECTION ----------------------- #
    def _is_extremum_3x3x3(self, prev_img: np.ndarray, curr_img: np.ndarray, next_img: np.ndarray, y: int, x: int) -> bool:
        val = curr_img[y, x]
        block = np.stack([prev_img[y - 1 : y + 2, x - 1 : x + 2],
                          curr_img[y - 1 : y + 2, x - 1 : x + 2],
                          next_img[y - 1 : y + 2, x - 1 : x + 2]], axis=0)
        if val == block.max():
            return np.count_nonzero(block == val) == 1
        if val == block.min():
            return np.count_nonzero(block == val) == 1
        return False

    def _edge_response_filter(self, dog: np.ndarray, y: int, x: int, r: float) -> bool:
        """
        Lọc điểm biên sử dụng điều kiện (Tr(H))^2 / Det(H) < (r+1)^2 / r
        với H là Hessian xấp xỉ trên DoG.
        Trả về True nếu VƯỢT NGƯỠNG (nên LOẠI), False nếu hợp lệ.
        """
        Dxx = dog[y, x + 1] + dog[y, x - 1] - 2 * dog[y, x]
        Dyy = dog[y + 1, x] + dog[y - 1, x] - 2 * dog[y, x]
        Dxy = (dog[y + 1, x + 1] - dog[y + 1, x - 1] - dog[y - 1, x + 1] + dog[y - 1, x - 1]) * 0.25
        tr = Dxx + Dyy
        det = Dxx * Dyy - Dxy * Dxy
        if det <= 1e-12:
            return True
        ratio = (tr * tr) / det
        r_thresh = ((r + 1) ** 2) / r
        return ratio > r_thresh

    def _find_scale_space_extrema(
        self,
        dog_pyr: List[List[np.ndarray]],
        sigmas_per_level: List[List[float]],
    ) -> List[Tuple[int, int, int, float]]:
        """
        Tìm điểm cực trị trong DoG pyramid.
        Trả về danh sách (o, s, y, x) kèm scale tuyệt đối (sigma) tại level s (lưu qua sigmas_per_level).
        """
        keypoints: List[Tuple[int, int, int, float]] = []

        # Ngưỡng tương phản: DoG theo ảnh [0,1] nên đặt thấp
        contrast_thresh = 0.5 * self.contrast_threshold / self.n_octave_layers

        for o, dogs in enumerate(dog_pyr):
            # dogs có (num_scales - 1) ảnh: s = 0..num_scales-2
            for s in range(1, len(dogs) - 1):
                prev_img = dogs[s - 1]
                curr_img = dogs[s]
                next_img = dogs[s + 1]

                h, w = curr_img.shape
                for y in range(1, h - 1):
                    # Có thể thêm mask ở đây nếu cần
                    for x in range(1, w - 1):
                        val = curr_img[y, x]
                        if abs(val) < contrast_thresh:
                            continue
                        if not self._is_extremum_3x3x3(prev_img, curr_img, next_img, y, x):
                            continue
                        # Lọc biên
                        if self._edge_response_filter(curr_img, y, x, r=self.edge_threshold):
                            continue
                        sigma_abs = sigmas_per_level[o][s]
                        keypoints.append((o, s, y, x, sigma_abs))

        return keypoints

    # ----------------------- ORIENTATION ASSIGNMENT ----------------------- #
    def _assign_orientations(
        self,
        gauss_pyr: List[List[np.ndarray]],
        raw_kps: List[Tuple[int, int, int, int, float]],
        num_bins: int = 36,
    ) -> List[cv2.KeyPoint]:
        """
        Với mỗi keypoint, tính histogram hướng trong vùng lân cận.
        Đỉnh lớn nhất -> hướng chính;.
        """
        keypoints: List[cv2.KeyPoint] = []
        bin_width = 360.0 / num_bins

        for o, s, y, x, sigma_abs in raw_kps:
            img = gauss_pyr[o][s]
            h, w = img.shape

            radius = int(round(3 * sigma_abs))
            if radius < 1:
                continue

            y0, y1 = max(1, y - radius), min(h - 1, y + radius + 1)
            x0, x1 = max(1, x - radius), min(w - 1, x + radius + 1)
            if y1 - y0 < 2 or x1 - x0 < 2:
                continue

            patch = img[y0:y1, x0:x1]
            # Gradient
            dx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=1)
            dy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=1)
            mag = np.sqrt(dx * dx + dy * dy)
            ori = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

            # Trọng số Gaussian
            yy, xx = np.mgrid[y0:y1, x0:x1]
            yy = yy - y
            xx = xx - x
            gauss_w = np.exp(-(xx * xx + yy * yy) / (2 * (1.5 * sigma_abs) ** 2)).astype(np.float32)

            hist = np.zeros((num_bins,), dtype=np.float32)
            for j in range(patch.shape[0]):
                for i in range(patch.shape[1]):
                    b = int(np.floor(ori[j, i] / bin_width)) % num_bins
                    hist[b] += mag[j, i] * gauss_w[j, i]

            # Làm mượt histogram (cửa sổ 3)
            hist = np.convolve(np.r_[hist[-1], hist, hist[0]], [1 / 3, 1 / 3, 1 / 3], mode="same")[1:-1]

            max_v = hist.max()
            if max_v <= 1e-6:
                continue

            # Các đỉnh >= 0.8*max_v
            for b, v in enumerate(hist):
                if v >= 0.8 * max_v:
                    angle = (b + 0.5) * bin_width
                    scale_factor = 2.0 ** o  # quy đổi toạ độ về ảnh gốc
                    # Dùng tham số vị trí để tương thích các phiên bản OpenCV:
                    # cv2.KeyPoint(x, y, size, angle, response, octave, class_id)
                    kp = cv2.KeyPoint(
                        float(x * scale_factor),
                        float(y * scale_factor),
                        float(2.0 * sigma_abs * scale_factor),
                        float(angle),
                        float(max_v),
                        int(o),
                        -1,
                    )
                    keypoints.append(kp)

        return keypoints

    # ----------------------- DESCRIPTOR 128-D ----------------------- #
    def _compute_descriptors(
        self,
        gauss_pyr: List[List[np.ndarray]],
        keypoints: List[cv2.KeyPoint],
        num_spatial_bins: int = 4,
        num_orientation_bins: int = 8,
        descriptor_max_val: float = 0.2,
    ) -> np.ndarray:
        """
        Tính descriptor 4x4 ô, mỗi ô 8 bins -> 128 chiều.
        - Vùng mẫu ~ 16x16 (tỷ lệ theo sigma/octave), có xoay theo hướng kp.angle.
        - Trọng số Gaussian và nội suy bilinear đơn giản vào histogram.
        """
        descs: List[np.ndarray] = []

        # Thông số mô tả
        bin_width = 360.0 / num_orientation_bins
        window_width = 16  # 16x16
        half_win = window_width // 2
        grid_size = window_width // num_spatial_bins  # 4

        for kp in keypoints:
            o = max(0, int(kp.octave))
            # Với mỗi keypoint, dùng level giữa của octave để đo gradient ổn định
            s = min(self.n_octave_layers, self.n_octave_layers)  # lấy 1 level hợp lý
            img = gauss_pyr[o][s]
            h, w = img.shape

            # Toạ độ về không gian octave hiện tại
            scale_factor = 2.0 ** o
            cx = kp.pt[0] / scale_factor
            cy = kp.pt[1] / scale_factor

            angle = (kp.angle % 360.0) * math.pi / 180.0
            cos_t = math.cos(angle)
            sin_t = math.sin(angle)

            # Tạo histogram rỗng (4x4x8)
            hist = np.zeros((num_spatial_bins, num_spatial_bins, num_orientation_bins), dtype=np.float32)

            # Cửa sổ 16x16 xoay quanh (cx, cy)
            for dy in range(-half_win, half_win):
                for dx in range(-half_win, half_win):
                    # Xoay điểm (dx, dy) -> (rx, ry)
                    rx = (dx * cos_t - dy * sin_t)
                    ry = (dx * sin_t + dy * cos_t)
                    x = int(round(cx + rx))
                    y = int(round(cy + ry))

                    if y <= 0 or y >= h - 1 or x <= 0 or x >= w - 1:
                        continue

                    # Gradient tại (y, x)
                    gx = float(img[y, x + 1] - img[y, x - 1])
                    gy = float(img[y + 1, x] - img[y - 1, x])
                    mag = math.sqrt(gx * gx + gy * gy)
                    ori = (math.degrees(math.atan2(gy, gx)) - kp.angle + 360.0) % 360.0

                    # Trọng số Gaussian giảm theo khoảng cách tới tâm
                    ww = math.exp(-(rx * rx + ry * ry) / (2 * (0.5 * window_width) ** 2))

                    # Vị trí ô (4x4)
                    bin_x = int((dx + half_win) / grid_size)
                    bin_y = int((dy + half_win) / grid_size)
                    if 0 <= bin_x < num_spatial_bins and 0 <= bin_y < num_spatial_bins:
                        obin = int(ori // bin_width) % num_orientation_bins
                        hist[bin_y, bin_x, obin] += ww * mag

            # Vector hoá 128 chiều
            vec = hist.flatten()
            # Chuẩn hoá L2
            norm = np.linalg.norm(vec) + 1e-12
            vec = vec / norm
            # Cắt ngưỡng và chuẩn hoá lại
            vec = np.clip(vec, 0, descriptor_max_val)
            norm = np.linalg.norm(vec) + 1e-12
            vec = (vec / norm).astype(np.float32)
            descs.append(vec)

        if not descs:
            return np.empty((0, 128), dtype=np.float32)
        return np.vstack(descs).astype(np.float32)


