"""
Image preprocessing module for the phishing detection pipeline.

Applies a sequence of image processing steps to prepare screenshots
for OCR text extraction. The pipeline:
    1. Decode raw bytes → OpenCV image
    2. Convert to grayscale
    3. Smart resize via `resize_image_for_ocr` (canvas-boundary logic)
    4. Denoise (fast non-local means)
    5. Contrast enhancement (CLAHE)
    6. Binarize (adaptive thresholding)

Utility:
    `scale_small_text_regions` — detects and upscales tiny text ROIs
    for improved Tesseract accuracy on dense/fine-print content.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

# ── Configuration ────────────────────────────────────────────────────
MIN_DIMENSION: int = 600    # Shortest side minimum (upscale threshold)
MAX_DIMENSION: int = 1920   # Longest side maximum  (downscale threshold)
DENOISE_H: int = 10         # Filter strength for fast NL-means denoising
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_SIZE: Tuple[int, int] = (8, 8)
ADAPTIVE_BLOCK_SIZE: int = 15   # Must be odd
ADAPTIVE_C: int = 8             # Constant subtracted from mean


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Full preprocessing pipeline: raw bytes → binarized image ready for OCR.

    Args:
        image_bytes: Raw image file content (PNG, JPEG, or WebP).

    Returns:
        Preprocessed image as a numpy array (uint8, single-channel).

    Raises:
        ValueError: If the image cannot be decoded.
    """
    # Step 1: Decode raw bytes into an OpenCV image
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image from provided bytes.")

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Smart resize — canvas boundary + aspect-ratio preservation
    gray = resize_image_for_ocr(gray, min_dim=MIN_DIMENSION, max_dim=MAX_DIMENSION)

    # Step 4: Denoise (operates on grayscale, before binarization)
    denoised = cv2.fastNlMeansDenoising(gray, h=DENOISE_H)

    # Step 5: Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_SIZE,
    )
    enhanced = clahe.apply(denoised)

    # Step 6: Adaptive threshold binarization
    binarized = cv2.adaptiveThreshold(
        enhanced,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=ADAPTIVE_BLOCK_SIZE,
        C=ADAPTIVE_C,
    )

    return binarized


def resize_image_for_ocr(
    gray_image: np.ndarray,
    min_dim: int = 600,
    max_dim: int = 1920,
) -> np.ndarray:
    """
    Proportionally resize a grayscale image to fit within a canvas defined
    by ``min_dim`` and ``max_dim``, preserving the original aspect ratio.

    Three mutually exclusive conditions are evaluated in order:

    * **Condition A — Downscale**: ``longest_side > max_dim``
        Scale so that the longest side equals ``max_dim``.
        Uses ``cv2.INTER_AREA`` (best quality for shrinking).

    * **Condition B — Upscale**: ``shortest_side < min_dim``
        Scale so that the shortest side equals ``min_dim``.
        Uses ``cv2.INTER_CUBIC`` (best quality for enlarging).

    * **Condition C — Passthrough**: ``min_dim <= shortest_side`` **and**
        ``longest_side <= max_dim``
        Image already fits the canvas; returned untouched.

    Resizing is always performed on the raw grayscale matrix — never on a
    binary image — to avoid anti-aliasing artifacts on binarized edges.

    Args:
        gray_image: Single-channel 2D NumPy array (``dtype=uint8``).
        min_dim:    Minimum allowed value for the shortest side (default 600).
        max_dim:    Maximum allowed value for the longest side (default 1920).

    Returns:
        Resized (or original) single-channel NumPy array.

    Raises:
        ValueError: If ``gray_image`` is not a 2-D array.
    """
    if gray_image.ndim != 2:
        raise ValueError(
            f"resize_image_for_ocr expects a 2-D grayscale array, "
            f"got shape {gray_image.shape}."
        )

    h: int
    w: int
    h, w = gray_image.shape

    longest_side: int = max(h, w)
    shortest_side: int = min(h, w)

    # ── Condition A: Downscale ────────────────────────────────────────
    if longest_side > max_dim:
        scale: float = max_dim / longest_side
        new_w: int = int(w * scale)
        new_h: int = int(h * scale)
        return cv2.resize(
            gray_image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

    # ── Condition B: Upscale ──────────────────────────────────────────
    if shortest_side < min_dim:
        scale = min_dim / shortest_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(
            gray_image,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC,
        )

    # ── Condition C: Passthrough ──────────────────────────────────────
    return gray_image


def scale_small_text_regions(
    gray_image: np.ndarray,
    min_box_height: int = 20,
) -> list[np.ndarray]:
    """
    Detect text bounding boxes via morphological operations and upscale any
    ROI whose height is below ``min_box_height``.

    Useful for screenshots that mix normal-size text with fine-print or
    caption text — Tesseract struggles with character heights below ~20px.

    Algorithm:
        1. Apply ``cv2.MORPH_CLOSE`` with a wide rectangular kernel ``(15, 3)``
           to merge horizontally adjacent glyphs into word/line blobs.
        2. Threshold the closed image to binary.
        3. Find external contours of the blobs.
        4. For each contour bounding box where ``h < min_box_height``
           and ``w > 15``, crop the original grayscale ROI and upscale
           it with ``cv2.INTER_CUBIC`` until its height equals
           ``min_box_height``.

    Args:
        gray_image:    Single-channel 2D NumPy array (``dtype=uint8``).
        min_box_height: Pixel height below which an ROI is considered
                        "small text" and will be upscaled (default 20).

    Returns:
        List of upscaled ROI arrays (``np.ndarray``). Empty if no small
        text regions are detected. Original-size ROIs are not included.

    Raises:
        ValueError: If ``gray_image`` is not a 2-D array.
    """
    if gray_image.ndim != 2:
        raise ValueError(
            f"scale_small_text_regions expects a 2-D grayscale array, "
            f"got shape {gray_image.shape}."
        )

    img_h: int
    img_w: int
    img_h, img_w = gray_image.shape

    # ── Step 1: Morphological close to merge glyph blobs ─────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

    # ── Step 2: Binarize the closed image for contour detection ───────
    _, binary = cv2.threshold(
        closed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ── Step 3: Find external contours ────────────────────────────────
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    scaled_rois: list[np.ndarray] = []

    # ── Step 4: Upscale small-height ROIs ─────────────────────────────
    for contour in contours:
        x: int
        y: int
        box_w: int
        box_h: int
        x, y, box_w, box_h = cv2.boundingRect(contour)

        # Skip boxes that are too tall (already fine) or too narrow (noise)
        if box_h >= min_box_height or box_w <= 15:
            continue

        # Guard against out-of-bounds slicing
        x2: int = min(x + box_w, img_w)
        y2: int = min(y + box_h, img_h)
        roi: np.ndarray = gray_image[y:y2, x:x2]

        if roi.size == 0:
            continue

        # Upscale proportionally until height == min_box_height
        scale: float = min_box_height / box_h
        new_roi_w: int = max(1, int(box_w * scale))
        scaled_roi: np.ndarray = cv2.resize(
            roi,
            (new_roi_w, min_box_height),
            interpolation=cv2.INTER_CUBIC,
        )
        scaled_rois.append(scaled_roi)

    return scaled_rois
