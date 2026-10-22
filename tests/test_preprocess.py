"""
Unit tests for image preprocessing and dual-variant OCR module.
"""

import io
import numpy as np
import cv2
from PIL import Image
import pytest

from src.preprocess import decode_image_bytes, resize_image_clamped, generate_preprocessing_variants
from src.ocr import check_unreadable_gate, compute_character_error_rate, extract_text_and_confidence


def test_decode_valid_and_invalid_bytes():
    # Valid PNG image bytes
    img = Image.new("RGB", (100, 200), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    valid_bytes = buf.getvalue()

    bgr = decode_image_bytes(valid_bytes)
    assert isinstance(bgr, np.ndarray)
    assert bgr.shape == (200, 100, 3)

    # Invalid bytes
    with pytest.raises(ValueError):
        decode_image_bytes(b"not an image data")

    with pytest.raises(ValueError):
        decode_image_bytes(b"")


def test_resize_image_clamped():
    # Test upscaling small image (< 600px shortest side)
    small_img = np.zeros((300, 200, 3), dtype=np.uint8)  # shortest = 200
    resized, scale = resize_image_clamped(small_img, min_side=600, max_side=1920)
    assert min(resized.shape[:2]) == 600
    assert scale == 3.0  # 600 / 200

    # Test downscaling large image (> 1920px longest side)
    large_img = np.zeros((2400, 1080, 3), dtype=np.uint8)  # longest = 2400
    resized2, scale2 = resize_image_clamped(large_img, min_side=600, max_side=1920)
    assert max(resized2.shape[:2]) == 1920
    assert scale2 == 1920 / 2400


def test_generate_preprocessing_variants():
    img = np.full((400, 400, 3), 120, dtype=np.uint8)
    # Add some high-contrast text-like pattern
    cv2.putText(img, "TEST", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    var_a, var_b = generate_preprocessing_variants(img)
    assert var_a.shape == (400, 400)
    assert var_b.shape == (400, 400)
    # Variant B is binary (values 0 or 255)
    unique_b = np.unique(var_b)
    assert all(val in [0, 255] for val in unique_b)


def test_check_unreadable_gate():
    # Good text and high confidence
    unreadable, reason = check_unreadable_gate("Your account has been restricted. Click here to verify.", 88.5)
    assert unreadable is False
    assert reason is None

    # Text too short (< 15 chars)
    unreadable, reason = check_unreadable_gate("Help!", 90.0, min_text_len=15)
    assert unreadable is True
    assert "too short" in reason

    # Confidence too low (< 35%)
    unreadable, reason = check_unreadable_gate("This text has enough length but low confidence score.", 28.0, min_confidence=35.0)
    assert unreadable is True
    assert "confidence too low" in reason


def test_compute_character_error_rate():
    ref = "Your verification code is 492018"
    assert compute_character_error_rate(ref, ref) == 0.0

    hyp_minor = "Your verifcation code is 492018"  # 1 deletion ('i')
    cer = compute_character_error_rate(ref, hyp_minor)
    assert 0.0 < cer < 0.10

    hyp_empty = ""
    assert compute_character_error_rate(ref, hyp_empty) == 1.0
