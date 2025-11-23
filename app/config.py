"""
Application configuration module.

Why:
- Centralize configuration for maintainability and testability.
- Use absolute paths to avoid issues with changing working directories.
"""

import os
from typing import List


class Config:
    """Base configuration for the Flask application."""

    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Đường dẫn thư mục giao diện tĩnh và template (cho Flask)
    TEMPLATES_DIR: str = os.path.join(BASE_DIR, "templates")
    STATIC_DIR: str = os.path.join(BASE_DIR, "static")

    # Thư mục lưu tạm file tải lên và ảnh kết quả
    UPLOAD_FOLDER: str = os.path.join(BASE_DIR, "uploads")
    RESULT_FOLDER: str = os.path.join(BASE_DIR, "results")

    # Giới hạn kích thước upload và định dạng cho phép
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS: List[str] = ["png", "jpg", "jpeg"]

    # CORS
    CORS_ORIGINS: List[str] = ["http://127.0.0.1:5500", "http://localhost:5500"]
    # Cổng chạy server Flask và chế độ debug
    PORT: int = 5000
    DEBUG: bool = True


