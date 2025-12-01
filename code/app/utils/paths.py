"""
Path utilities.

Why:
- Centralize directory creation helpers to reduce duplication and improve testability.
"""

import os


def ensure_dirs(*paths: str) -> None:
    """
    Ensure target directories exist (idempotent).

    Rationale:
    - Avoid runtime errors when writing uploads/results if directories are missing.
    """
    # Tạo các thư mục mục tiêu nếu chưa tồn tại để tránh lỗi khi lưu file
    for path in paths:
        os.makedirs(path, exist_ok=True)


