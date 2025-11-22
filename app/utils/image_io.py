"""
Image I/O utilities.

Why:
- Separate image encoding logic from API layer for reuse and testability.
"""

import base64
from typing import Optional

import cv2
import numpy as np


def encode_image_to_base64(img_bgr: np.ndarray, ext: str = ".jpg") -> Optional[str]:
    """
    Encode a BGR image (OpenCV) to a base64 data URL string.

    Rationale:
    - Frontend can display the result immediately without relying on filesystem paths.
    """
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        return None
    b64 = base64.b64encode(buf).decode("utf-8")
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


