"""
HTTP routes (Blueprint) for the application.

Why:
- Keep HTTP transport separate from business logic (SRP).
- Easier to unit test services without Flask.
"""

import os
from typing import List

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, render_template, request, send_file

from app.services.panorama_service import PanoramaService
from app.sift import SIFT
from app.utils.image_io import encode_image_to_base64
from app.utils.paths import ensure_dirs

bp = Blueprint("web", __name__)


def _allowed_file(filename: str) -> bool:
    """Check filename extension against configured allowed list."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]


@bp.get("/")
def index():
    """Render home page."""
    # Trang chủ giao diện Flask (hiện không cần khi chạy Streamlit)
    return render_template("index.html")


@bp.post("/stitch")
def stitch_images():
    """Stitch uploaded images into a panorama and return JSON with base64 image."""
    # API: nhận ảnh tải lên -> decode -> ghép panorama bằng PanoramaService -> lưu & trả JSON
    try:
        if "images" not in request.files:
            return jsonify({"error": "Không có file nào được tải lên"}), 400

        files = request.files.getlist("images")
        if len(files) < 2:
            return jsonify({"error": "Cần ít nhất 2 ảnh để ghép panorama"}), 400

        images: List[np.ndarray] = []
        for file in files:
            if file and _allowed_file(file.filename):
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                max_dim = 800
                h, w = img.shape[:2]
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    img = cv2.resize(img, None, fx=scale, fy=scale)
                images.append(img)

        if len(images) < 2:
            return jsonify({"error": "Không đủ ảnh hợp lệ để ghép"}), 400

        ensure_dirs(current_app.config["RESULT_FOLDER"], current_app.config["UPLOAD_FOLDER"])

        service = PanoramaService(
            sift=SIFT(n_octave_layers=3, contrast_threshold=0.04, edge_threshold=10, sigma=1.6)
        )
        panorama = service.stitch(images)

        result_id = os.urandom(8).hex()
        result_path = os.path.join(current_app.config["RESULT_FOLDER"], f"{result_id}.jpg")
        cv2.imwrite(result_path, panorama)

        image_data = encode_image_to_base64(panorama, ".jpg")

        return jsonify(
            {
                "success": True,
                "result_id": result_id,
                "message": f"Đã ghép thành công {len(images)} ảnh!",
                "image_data": image_data,
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Lỗi: {str(exc)}"}), 500


@bp.get("/result/<result_id>")
def get_result(result_id: str):
    """Return saved panorama file for download or fallback display."""
    # API: trả file ảnh kết quả theo mã id (download/hiển thị)
    try:
        result_path = os.path.join(current_app.config["RESULT_FOLDER"], f"{result_id}.jpg")
        if os.path.exists(result_path):
            return send_file(result_path, mimetype="image/jpeg")
        return jsonify({"error": "Không tìm thấy ảnh"}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


