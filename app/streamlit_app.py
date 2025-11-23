import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st


# Ensure project root is on sys.path when running from inside the package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.services.panorama_service import PanoramaService
from app.sift import SIFT


def decode_image_file_to_bgr(file_bytes: bytes, max_dim: int = 800) -> np.ndarray:
    """
    Decode raw bytes to BGR image (OpenCV) and resize if larger than max_dim.
    """
    file_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if img is None:
        return img
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    return img


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def encode_jpeg(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if not ok:
        return b""
    return buf.tobytes()


def main() -> None:
    st.set_page_config(page_title="Ghép Ảnh", page_icon="🖼️", layout="wide")

    # Center content by constraining max width and using a middle column
    st.markdown(
        """
        <style>
        .block-container {max-width: 1100px; margin: 0 auto;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    _left, _center, _right = st.columns([1, 6, 1])
    with _center:
        st.title("🖼️ Ghép Ảnh")
        st.write("Ghép nhiều ảnh thành một bức tranh toàn cảnh")

        # Upload
        uploaded_files = st.file_uploader(
            "Kéo thả ảnh vào đây hoặc click để chọn (JPG/PNG, tối thiểu 2 ảnh)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        kept_files: List[Tuple[str, bytes]] = []

        if uploaded_files:
            st.subheader(f"Ảnh đã chọn ({len(uploaded_files)})")
            cols = st.columns(min(4, max(1, len(uploaded_files))))
            for i, f in enumerate(uploaded_files):
                with cols[i % len(cols)]:
                    file_bytes = f.getvalue()
                    # Show preview smaller (convert to RGB for Streamlit)
                    img_bgr = decode_image_file_to_bgr(file_bytes, max_dim=300)
                    if img_bgr is not None:
                        st.image(bgr_to_rgb(img_bgr), caption=f.name, width=220)
                    remove = st.checkbox("Bỏ ảnh này", key=f"remove_{i}", value=False)
                    if not remove:
                        kept_files.append((f.name, file_bytes))

        stitch_disabled = len(kept_files) < 2
        if st.button("🎨 Ghép Ảnh", type="primary", disabled=stitch_disabled):
            if len(kept_files) < 2:
                st.error("Vui lòng chọn ít nhất 2 ảnh hợp lệ!")
                return

            with st.spinner("Đang xử lý ảnh... Vui lòng đợi"):
                # Decode images
                images: List[np.ndarray] = []
                for name, data in kept_files:
                    img = decode_image_file_to_bgr(data, max_dim=800)
                    if img is not None:
                        images.append(img)

                if len(images) < 2:
                    st.error("Không đủ ảnh hợp lệ để ghép.")
                    return

                # Panorama pipeline
                service = PanoramaService(
                    sift=SIFT(n_octave_layers=3, contrast_threshold=0.04, edge_threshold=10, sigma=1.6)
                )
                try:
                    panorama = service.stitch(images)
                except Exception as exc:
                    st.error(f"Lỗi khi ghép ảnh: {exc}")
                    return

            st.success(f"Đã ghép thành công {len(images)} ảnh!")
            st.image(bgr_to_rgb(panorama), caption="✨ Kết Quả Panorama", width="stretch")
            jpeg_bytes = encode_jpeg(panorama)
            if jpeg_bytes:
                st.download_button(
                    "⬇️ Tải Xuống Ảnh",
                    data=jpeg_bytes,
                    file_name="panorama.jpg",
                    mime="image/jpeg",
                )
            else:
                st.warning("Không thể mã hoá ảnh đầu ra để tải xuống.")


if __name__ == "__main__":
    main()

