"""Offline Mathematical OCR Engine.

Recognizes mathematical equations and expressions from images (photos, screenshots)
100% locally on CPU without any cloud APIs or external API keys.

Uses RapidOCR (ONNX Runtime) with fallback to native Windows Media OCR (winocr).
Includes domain-specific mathematical post-processing.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

logger = logging.getLogger(__name__)


class LocalMathOCR:
    """Offline OCR processor for mathematical equations."""

    _rapid_engine = None

    @classmethod
    def get_rapid_engine(cls):
        """Lazily initialize RapidOCR engine."""
        if cls._rapid_engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                cls._rapid_engine = RapidOCR()
            except Exception as e:
                logger.warning("RapidOCR unavailable: %s", e)
                cls._rapid_engine = False
        return cls._rapid_engine if cls._rapid_engine is not False else None

    @classmethod
    def preprocess_image(cls, img: Image.Image) -> Image.Image:
        """Preprocess image for optimal mathematical character recognition."""
        # Convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Upscale if image is small
        min_dim = min(img.width, img.height)
        if min_dim < 300:
            scale = max(2.0, 400.0 / max(min_dim, 1))
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.4)

        return img

    @classmethod
    def clean_math_text(cls, text: str) -> str:
        """Sanitize and standardize raw OCR output into clean mathematical notation."""
        if not text:
            return ""

        s = text.strip()

        # Unicode arithmetic replacements
        s = s.replace("–", "-").replace("—", "-").replace("−", "-")
        s = s.replace("×", "*").replace("·", "*").replace("÷", "/")
        s = s.replace("√", "sqrt")

        # Superscript conversions
        super_map = {
            "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4",
            "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
            "⁺": "+", "⁻": "-",
        }
        for k, v in super_map.items():
            s = s.replace(k, v)

        # Common OCR misidentifications in math formulas
        # e.g., 'xA2' -> 'x^2'
        s = re.sub(r"([a-zA-Z])A(\d+)", r"\1^\2", s)
        # e.g. 'y =' or 'f(x) =' spacing
        s = re.sub(r"\s+", " ", s).strip()

        return s

    @classmethod
    def recognize(cls, image: Image.Image) -> Tuple[str, str]:
        """Perform local offline OCR on image.

        Returns:
            Tuple of (extracted_text, engine_used)
        """
        processed_img = cls.preprocess_image(image)

        # 1. Try RapidOCR (ONNX)
        rapid = cls.get_rapid_engine()
        if rapid:
            try:
                np_img = np.array(processed_img)
                results, _ = rapid(np_img)
                if results:
                    # Sort detected text boxes: primarily by top coordinate (line), then left coordinate
                    # Box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    sorted_boxes = sorted(
                        results,
                        key=lambda r: (round(min(pt[1] for pt in r[0]) / 35.0), min(pt[0] for pt in r[0]))
                    )
                    raw_text = " ".join([b[1] for b in sorted_boxes])
                    cleaned = cls.clean_math_text(raw_text)
                    if cleaned:
                        return cleaned, "RapidOCR (Local CPU ONNX)"
            except Exception as e:
                logger.debug("RapidOCR extraction error: %s", e)

        # 2. Try Windows Media OCR (winocr)
        try:
            import winocr
            rgba_img = processed_img.convert("RGBA")
            res = winocr.recognize_pil_sync(rgba_img, lang="en-US")
            text = res.get("text", "")
            if not text:
                res = winocr.recognize_pil_sync(rgba_img, lang="en-GB")
                text = res.get("text", "")
            if text:
                cleaned = cls.clean_math_text(text)
                return cleaned, "Windows Media OCR (Local OS Engine)"
        except Exception as e:
            logger.debug("WinOCR fallback error: %s", e)

        return "", "No text detected"
