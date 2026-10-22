"""
Image preprocessing module for phishing screenshot detection.

Handles byte decoding, dimension clamping (600px - 1920px), EXIF stripping,
and dual-variant image enhancement (CLAHE vs NLM-Denoised Adaptive Gaussian Threshold).
"""

import io
from typing import Tuple, Union, Optional
import numpy as np
import cv2
from PIL import Image, ImageOps


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes into a BGR color OpenCV array.
    Strips EXIF orientation in memory. Raises ValueError on undecodable bytes.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Empty image bytes received")

    try:
        # Load with PIL first to automatically handle EXIF orientation transpose
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        # Convert to RGB (handles RGBA, Palette, Grayscale)
        rgb_img = pil_img.convert("RGB")
        img_np = np.array(rgb_img)
        # Convert RGB to BGR for OpenCV consistency
        bgr_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return bgr_img
    except Exception as e:
        raise ValueError(f"Undecodable image: {e}") from e


def resize_image_clamped(
    img: np.ndarray,
    min_side: int = 600,
    max_side: int = 1920
) -> Tuple[np.ndarray, float]:
    """
    Resize image so that:
      - shortest side >= min_side (upscale using INTER_CUBIC)
      - longest side <= max_side (downscale using INTER_AREA)
    Preserves exact aspect ratio.
    
    Returns:
        (resized_image, scale_factor)
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Image has zero dimension")

    shortest = min(h, w)
    longest = max(h, w)

    scale = 1.0

    if longest > max_side:
        scale = max_side / longest
    elif shortest < min_side:
        scale = min_side / shortest

    if abs(scale - 1.0) < 1e-3:
        return img.copy(), 1.0

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    return resized, scale


def generate_preprocessing_variants(
    bgr_img: np.ndarray,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    nl_means_h: int = 10,
    adaptive_block_size: int = 15,
    adaptive_c: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two preprocessed variants optimized for OCR:
      Variant A: Grayscale + CLAHE (enhances contrast on low-contrast/dark-mode screens).
      Variant B: Grayscale + Non-Local Means Denoise + Adaptive Gaussian Threshold (binarized).

    Returns:
        (variant_a_clahe, variant_b_adaptive_threshold)
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # Variant A: CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    variant_a = clahe.apply(gray)

    # Variant B: Fast NLM Denoising + Adaptive Gaussian Threshold
    denoised = cv2.fastNlMeansDenoising(gray, h=nl_means_h)
    # Ensure block size is odd and >= 3
    block_size = adaptive_block_size if (adaptive_block_size % 2 == 1) else (adaptive_block_size + 1)
    variant_b = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        adaptive_c
    )

    return variant_a, variant_b


def crop_phone_system_bars(
    img: np.ndarray,
    aspect_ratio_threshold: float = 1.6,
    top_bar_pct: float = 0.045,
    bottom_bar_pct: float = 0.045,
) -> Tuple[np.ndarray, bool]:
    """
    Geometrically crop status bar (top) and navigation bar (bottom) on phone-size images.

    Args:
        img: Input BGR or grayscale image array.
        aspect_ratio_threshold: Minimum height/width ratio to consider an image phone-sized (default 1.6).
        top_bar_pct: Fraction of height occupied by top status bar (default 0.045, ~108px on 2400h).
        bottom_bar_pct: Fraction of height occupied by bottom navigation bar (default 0.045).

    Returns:
        (cropped_image, was_cropped_boolean)
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img, False

    aspect_ratio = h / float(w)
    if aspect_ratio >= aspect_ratio_threshold:
        y_start = int(round(h * top_bar_pct))
        y_end = int(round(h * (1.0 - bottom_bar_pct)))
        if y_end > y_start + 50:
            return img[y_start:y_end, :].copy(), True

    return img, False
