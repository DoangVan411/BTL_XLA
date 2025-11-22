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


