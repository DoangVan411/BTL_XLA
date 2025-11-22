from typing import Tuple

import numpy as np


def perspective_transform(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Biến đổi phối cảnh (perspective) cho tập điểm 2D bằng homography H.
    - points: mảng toạ độ 2D dạng (N, 2) hoặc (N, 1, 2)
    - H: ma trận homography 3x3
    Trả về:
    - points_out: dạng (N, 1, 2) float32 để tương thích với cách dùng trong pipeline.
    """
    if points.ndim == 3:
        pts = points.reshape(-1, 2).astype(np.float64)
    else:
        pts = points.astype(np.float64)
    assert pts.shape[1] == 2, "points phải có shape (N, 2) hoặc (N, 1, 2)"
    assert H.shape == (3, 3), "H phải là ma trận 3x3"

    # Chuyển sang toạ độ đồng nhất
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])  # (N, 3)
    proj = (H @ pts_h.T).T  # (N, 3)
    w = proj[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12  # tránh chia cho 0
    xy = proj[:, :2] / w

    # Trả về (N, 1, 2) float32
    return xy.reshape(-1, 1, 2).astype(np.float32)

def _bilinear_sample(image: np.ndarray, x: float, y: float) -> np.ndarray:
    """
    Lấy mẫu bilinear tại toạ độ (x, y) trên ảnh (BGR).
    - image: HxWxC (uint8 hoặc float)
    - x, y: toạ độ float trong hệ ảnh (0..W-1, 0..H-1)
    Trả về vector C kênh (float32).
    """
    h, w = image.shape[:2]
    if x < 0 or x > w - 1 or y < 0 or y > h - 1:
        return np.zeros((image.shape[2] if image.ndim == 3 else 1,), dtype=np.float32)
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, w - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0
    if image.ndim == 2:
        Ia = float(image[y0, x0])
        Ib = float(image[y0, x1])
        Ic = float(image[y1, x0])
        Id = float(image[y1, x1])
        top = Ia * (1 - dx) + Ib * dx
        bottom = Ic * (1 - dx) + Id * dx
        val = (top * (1 - dy) + bottom * dy)
        return np.array([val], dtype=np.float32)
    else:
        Ia = image[y0, x0].astype(np.float32)
        Ib = image[y0, x1].astype(np.float32)
        Ic = image[y1, x0].astype(np.float32)
        Id = image[y1, x1].astype(np.float32)
        top = Ia * (1 - dx) + Ib * dx
        bottom = Ic * (1 - dx) + Id * dx
        return (top * (1 - dy) + bottom * dy).astype(np.float32)

def warp_perspective(image: np.ndarray, H: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    """
    Biến đổi phối cảnh ảnh (không dùng cv2.warpPerspective).
    - image: ảnh nguồn (H_src, W_src, C) BGR uint8
    - H: ma trận homography 3x3 (áp dụng theo hướng dst -> src sử dụng nghịch đảo khi lấy mẫu)
    - output_shape: (W_out, H_out) giống tham số dsize của OpenCV
    Thuật toán:
    - Duyệt từng điểm (x_d, y_d) trên ảnh đích
    - Chiếu ngược bằng H^{-1} -> (x_s, y_s) trên ảnh nguồn
    - Lấy mẫu bilinear từ ảnh nguồn, gán vào ảnh đích
    """
    W_out, H_out = output_shape
    H_inv = np.linalg.inv(H)

    # Chuẩn bị đầu ra
    if image.ndim == 2:
        dst = np.zeros((H_out, W_out), dtype=np.float32)
    else:
        C = image.shape[2]
        dst = np.zeros((H_out, W_out, C), dtype=np.float32)

    # Lặp qua từng pixel đích, inverse mapping về nguồn
    for y_d in range(H_out):
        for x_d in range(W_out):
            p = np.array([x_d, y_d, 1.0], dtype=np.float64)
            q = H_inv @ p
            w = q[2] if abs(q[2]) > 1e-12 else 1e-12
            x_s = q[0] / w
            y_s = q[1] / w
            val = _bilinear_sample(image, x_s, y_s)
            if image.ndim == 2:
                dst[y_d, x_d] = val[0]
            else:
                dst[y_d, x_d, :] = val

    # Trả về cùng kiểu với ảnh gốc (uint8)
    dst = np.clip(dst, 0, 255)
    return dst.astype(np.uint8)


