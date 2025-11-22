from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Match:
    """
    Cấu trúc tối giản thay cho cv2.DMatch để dùng trong pipeline:
    - queryIdx: chỉ số descriptor bên trái (desc1)
    - trainIdx: chỉ số descriptor bên phải (desc2)
    - distance: khoảng cách Euclid giữa hai descriptor
    """
    queryIdx: int
    trainIdx: int
    distance: float


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Tính ma trận khoảng cách Euclid bình phương giữa 2 tập descriptor:
      - a: (N1, D), b: (N2, D)
    Trả về: (N1, N2) khoảng cách bình phương.
    Công thức: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    """
    # Bảo vệ input
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return np.empty((0, 0), dtype=np.float32)
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    a2 = np.sum(a * a, axis=1, keepdims=True)        # (N1, 1)
    b2 = np.sum(b * b, axis=1, keepdims=True).T      # (1, N2)
    ab = a @ b.T                                     # (N1, N2)
    d2 = np.maximum(a2 + b2 - 2.0 * ab, 0.0)
    return d2


def knn_match(desc1: Optional[np.ndarray], desc2: Optional[np.ndarray], k: int = 2) -> List[List[Match]]:
    """
    KNN matching tự cài đặt (không dùng FLANN/BFMatcher):
    - Tính khoảng cách giữa mọi cặp descriptor bằng NumPy.
    - Với mỗi descriptor bên trái, lấy k láng giềng gần nhất bên phải.
    - Trả về danh sách các cặp match theo đúng format của knnMatch:
        matches = [ [Match_best, Match_second], [ ... ], ... ]
    Lưu ý: Nếu desc2 có ít hơn k phần tử, các hàng tương ứng sẽ ngắn hơn.
    """
    if desc1 is None or desc2 is None:
        return []
    if len(desc1) == 0 or len(desc2) == 0:
        return []

    k = max(1, int(k))
    # Nếu không đủ để lấy k láng giềng, vẫn trả về số có thể
    k_eff = min(k, len(desc2))

    # Ma trận khoảng cách bình phương
    d2 = _pairwise_distances(desc1, desc2)  # (N1, N2)
    if d2.size == 0:
        return []

    # Lấy k chỉ số nhỏ nhất theo từng hàng với argpartition (O(N))
    # Sau đó sắp xếp lại k chỉ số này theo thứ tự tăng dần khoảng cách
    idx_part = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]  # (N1, k_eff)
    rows = np.arange(d2.shape[0])[:, None]
    d2_part = d2[rows, idx_part]
    order = np.argsort(d2_part, axis=1)
    sorted_idx = idx_part[rows, order]  # (N1, k_eff)
    sorted_d2 = d2_part[rows, order]    # (N1, k_eff)

    # Tạo danh sách Match theo từng hàng
    matches: List[List[Match]] = []
    for i in range(sorted_idx.shape[0]):
        row_matches: List[Match] = []
        for j in range(sorted_idx.shape[1]):
            jj = int(sorted_idx[i, j])
            dist = float(np.sqrt(sorted_d2[i, j]))
            row_matches.append(Match(queryIdx=i, trainIdx=jj, distance=dist))
        matches.append(row_matches)
    return matches


