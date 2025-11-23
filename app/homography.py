from typing import Optional, Tuple

import numpy as np


def _smallest_singular_vector(
    A: np.ndarray, max_iters: int = 2000, tol: float = 1e-10, seed: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Tìm vector singular tương ứng singular value nhỏ nhất của A
    bằng power-iteration trên ma trận B = c*I - (A^T A),
    trong đó c >= lambda_max(A^T A) (dùng c = trace để đảm bảo).

    Trả về vector đơn vị v sao cho v ~ argmin ||A v||, hoặc None nếu thất bại.
    """
    if A.size == 0:
        return None
    # M = A^T A là đối xứng dương bán xác định
    M = A.T @ A
    n = M.shape[0]
    if n == 0:
        return None

    # Chọn c >= lambda_max(M). trace(M) >= lambda_max(M) luôn đúng.
    c = float(np.trace(M)) + 1e-12
    B = (c * np.eye(n, dtype=np.float64)) - M

    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n,)).astype(np.float64)
    # Tránh vector khởi tạo quá nhỏ
    nv = np.linalg.norm(v) + 1e-18
    v = v / nv

    last_v = v
    for _ in range(max_iters):
        w = B @ v
        nw = np.linalg.norm(w)
        if nw < 1e-20:
            v = rng.normal(size=(n,)).astype(np.float64)
            nv = np.linalg.norm(v) + 1e-18
            v = v / nv
            continue
        v = w / nw
        if np.linalg.norm(v - last_v) < tol:
            break
        last_v = v

    # Chuẩn hoá kết quả
    nv = np.linalg.norm(v)
    if nv < 1e-20 or not np.isfinite(nv):
        return None
    return (v / nv).astype(np.float64)


def _normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chuẩn hoá điểm (Hartley normalization) để tăng ổn định số:
    - Dịch tâm về gốc toạ độ
    - Scale sao cho khoảng cách trung bình tới gốc = sqrt(2)
    Input:
      pts: (N, 2) hoặc (N, 1, 2)
    Output:
      pts_norm: (N, 2)
      T: ma trận biến đổi 3x3
    """
    # Chuẩn hoá toạ độ giúp thuật toán DLT ổn định hơn về mặt số học
    if pts.ndim == 3:
        pts_2d = pts.reshape(-1, 2)
    else:
        pts_2d = pts

    centroid = np.mean(pts_2d, axis=0)
    shifted = pts_2d - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)) + 1e-12)
    scale = np.sqrt(2.0) / (mean_dist + 1e-12)

    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pts_h = np.hstack([pts_2d.astype(np.float64), np.ones((pts_2d.shape[0], 1), dtype=np.float64)])
    pts_norm_h = (T @ pts_h.T).T
    pts_norm = pts_norm_h[:, :2] / pts_norm_h[:, 2:3]
    return pts_norm, T


def _dlt_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
    """
    Tính ma trận homography H (3x3) sao cho dst ~ H * src bằng DLT với chuẩn hoá.
    - src_pts, dst_pts: (N, 2) hoặc (N, 1, 2), N >= 4
    Trả về H hoặc None nếu suy biến.
    """
    # Dựng hệ phương trình từ các cặp điểm, giải bằng SVD để suy ra H
    if src_pts.shape[0] < 4 or dst_pts.shape[0] < 4:
        return None

    # Chuẩn hoá
    src_n, T_src = _normalize_points(src_pts)
    dst_n, T_dst = _normalize_points(dst_pts)

    # Xây ma trận A từ tương ứng dst ~ H * src
    # Với mỗi (x, y) -> (u, v):
    # [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
    # [ 0,  0,  0,-x,-y,-1, v*x, v*y, v]
    N = src_n.shape[0]
    A = np.zeros((2 * N, 9), dtype=np.float64)
    x = src_n[:, 0]
    y = src_n[:, 1]
    u = dst_n[:, 0]
    v = dst_n[:, 1]

    A[0::2, 0:3] = np.stack([-x, -y, -np.ones_like(x)], axis=1)
    A[0::2, 6:9] = np.stack([u * x, u * y, u], axis=1)
    A[1::2, 3:6] = np.stack([-x, -y, -np.ones_like(x)], axis=1)
    A[1::2, 6:9] = np.stack([v * x, v * y, v], axis=1)

    # Giải vector singular nhỏ nhất bằng power-iteration trên B = cI - A^T A
    h = _smallest_singular_vector(A, max_iters=2000, tol=1e-12, seed=None)
    if h is None:
        return None
    Hn = h.reshape(3, 3)

    # Khử chuẩn hoá
    try:
        H = np.linalg.inv(T_dst) @ Hn @ T_src
    except np.linalg.LinAlgError:
        return None

    # Chuẩn hoá sao cho H[2,2] = 1
    if abs(H[2, 2]) < 1e-12:
        return None
    H = H / H[2, 2]
    return H


def _project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Chiếu điểm pts (N, 2) bằng H: (x, y, 1) -> (u, v, w) -> (u/w, v/w)
    """
    # Chiếu toạ độ bằng không gian đồng nhất và quy đổi về (x, y)
    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)
    pts_h = np.hstack([pts.astype(np.float64), np.ones((pts.shape[0], 1), dtype=np.float64)])
    proj = (H @ pts_h.T).T
    w = proj[:, 2:3] + 1e-12
    proj_xy = proj[:, :2] / w
    return proj_xy


def find_homography_ransac(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    ransac_reproj_threshold: float = 5.0,
    max_iters: int = 2000,
    seed: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Tìm homography với RANSAC (không dùng cv2.findHomography).
    - src_pts: (N, 2) điểm nguồn
    - dst_pts: (N, 2) điểm đích
    - ransac_reproj_threshold: ngưỡng inlier theo lỗi Euclid (pixel)
    - max_iters: số vòng lặp RANSAC
    Trả về:
      H (3x3) và mask (N, 1) với 1=inlier, 0=outlier; hoặc (None, None) nếu thất bại.
    """
    # Lặp chọn mẫu nhỏ (4 cặp), ước lượng H, đếm inlier theo lỗi chiếu; chọn H tốt nhất và tinh chỉnh
    if src_pts.shape[0] < 4 or dst_pts.shape[0] < 4:
        return None, None

    if src_pts.ndim == 3:
        src = src_pts.reshape(-1, 2)
    else:
        src = src_pts
    if dst_pts.ndim == 3:
        dst = dst_pts.reshape(-1, 2)
    else:
        dst = dst_pts

    N = src.shape[0]
    rng = np.random.default_rng(seed)

    best_H: Optional[np.ndarray] = None
    best_inliers: Optional[np.ndarray] = None
    best_count = 0

    # RANSAC: lặp chọn 4 cặp ngẫu nhiên, ước lượng H, chấm điểm inliers
    for _ in range(max_iters):
        sample_idx = rng.choice(N, size=4, replace=False)
        H = _dlt_homography(src[sample_idx], dst[sample_idx])
        if H is None:
            continue

        proj = _project_points(H, src)
        errors = np.sqrt(np.sum((proj - dst) ** 2, axis=1))
        inliers = errors < ransac_reproj_threshold
        count = int(np.count_nonzero(inliers))

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_H = H

            # Early exit nếu phần lớn là inliers
            if best_count > 0.8 * N:
                break

    if best_H is None or best_inliers is None or best_count < 4:
        return None, None

    # Tinh chỉnh H bằng tất cả inliers
    H_refined = _dlt_homography(src[best_inliers], dst[best_inliers])
    if H_refined is None:
        H_refined = best_H

    mask = best_inliers.astype(np.uint8).reshape(-1, 1)
    return H_refined, mask


