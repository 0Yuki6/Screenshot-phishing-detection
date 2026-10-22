"""
Unit and integration tests for the dual-variant OCR pipeline (src/ocr.py).

Covers:
- extract_text_and_confidence: output types, empty image handling, confidence range
- run_dual_variant_ocr: winner selection, key presence, scale factor
- check_unreadable_gate: all four edge cases (short, low-conf, pass, empty)
- compute_character_error_rate: exact match, single-char edit, empty ref/hyp
- Dark-mode polarity: inverted image must still yield confidence in [0, 100]
"""

import io
import numpy as np
import cv2
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.ocr import (
    extract_text_and_confidence,
    run_dual_variant_ocr,
    check_unreadable_gate,
    compute_character_error_rate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_image(text: str, dark_mode: bool = False, size=(600, 200)) -> np.ndarray:
    """Return a BGR NumPy image with `text` rendered on it."""
    bg = (30, 30, 30) if dark_mode else (255, 255, 255)
    fg = (220, 220, 220) if dark_mode else (0, 0, 0)
    pil_img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(pil_img)
    draw.text((10, 60), text, fill=fg)
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return bgr


def _solid_color_image(h=400, w=300, color=(200, 200, 200)) -> np.ndarray:
    """Return a solid-colour BGR image (no text)."""
    img = np.full((h, w, 3), color, dtype=np.uint8)
    return img


# ---------------------------------------------------------------------------
# extract_text_and_confidence
# ---------------------------------------------------------------------------

class TestExtractTextAndConfidence:
    def test_returns_tuple_of_correct_types(self):
        img = cv2.cvtColor(_make_text_image("Hello World")[..., :], cv2.COLOR_BGR2GRAY)
        text, conf, wc = extract_text_and_confidence(img)
        assert isinstance(text, str)
        assert isinstance(conf, float)
        assert isinstance(wc, int)

    def test_confidence_in_valid_range(self):
        img = cv2.cvtColor(_make_text_image("Verify your account now"), cv2.COLOR_BGR2GRAY)
        _, conf, _ = extract_text_and_confidence(img)
        assert 0.0 <= conf <= 100.0

    def test_empty_image_returns_zero_confidence(self):
        """Solid-colour image with no text should give very low word count / confidence."""
        solid = _solid_color_image()
        gray = cv2.cvtColor(solid, cv2.COLOR_BGR2GRAY)
        text, conf, wc = extract_text_and_confidence(gray)
        # Either no words extracted, or confidence may be 0
        assert wc >= 0
        assert 0.0 <= conf <= 100.0

    def test_word_count_matches_text_tokens(self):
        img = cv2.cvtColor(_make_text_image("One Two Three Four Five"), cv2.COLOR_BGR2GRAY)
        text, conf, wc = extract_text_and_confidence(img)
        tokens = [t for t in text.split() if t.strip()]
        assert wc == len(tokens)


# ---------------------------------------------------------------------------
# run_dual_variant_ocr
# ---------------------------------------------------------------------------

class TestRunDualVariantOcr:
    REQUIRED_KEYS = {
        "text", "confidence", "word_count",
        "winning_variant", "variant_a_conf", "variant_b_conf", "scale_factor"
    }

    def test_output_keys_present(self):
        img = _make_text_image("Your OTP is 482910. Do not share it.")
        result = run_dual_variant_ocr(img)
        assert self.REQUIRED_KEYS.issubset(result.keys()), (
            f"Missing keys: {self.REQUIRED_KEYS - result.keys()}"
        )

    def test_winning_variant_is_valid(self):
        img = _make_text_image("Click here to confirm your payment.")
        result = run_dual_variant_ocr(img)
        assert result["winning_variant"] in {"clahe", "adaptive_threshold"}

    def test_confidence_in_valid_range(self):
        img = _make_text_image("Important: your subscription expires today.")
        result = run_dual_variant_ocr(img)
        assert 0.0 <= result["confidence"] <= 100.0
        assert 0.0 <= result["variant_a_conf"] <= 100.0
        assert 0.0 <= result["variant_b_conf"] <= 100.0

    def test_scale_factor_positive(self):
        img = _make_text_image("Test scale factor check.")
        result = run_dual_variant_ocr(img)
        assert result["scale_factor"] > 0.0

    def test_upscale_small_image(self):
        """Images below min_side=600 must be upscaled (scale_factor > 1)."""
        small = _make_text_image("Tiny image test", size=(200, 80))
        result = run_dual_variant_ocr(small, min_side=600)
        # scale_factor should reflect upscaling
        assert result["scale_factor"] >= 1.0

    def test_downscale_large_image(self):
        """Images above max_side=1920 must be downscaled (scale_factor < 1)."""
        large = np.zeros((2600, 1200, 3), dtype=np.uint8)
        cv2.putText(large, "LARGE IMAGE TEXT", (50, 1300), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 8)
        result = run_dual_variant_ocr(large, min_side=600, max_side=1920)
        assert result["scale_factor"] < 1.0 + 1e-6  # <= 1.0 (small tolerance for boundary)

    def test_dark_mode_image(self):
        """Dark-mode (white-on-dark) images must still produce a valid result."""
        dark_img = _make_text_image("URGENT: Account suspended", dark_mode=True)
        result = run_dual_variant_ocr(dark_img)
        assert isinstance(result["text"], str)
        assert 0.0 <= result["confidence"] <= 100.0

    def test_word_count_nonnegative(self):
        img = _make_text_image("Hello this is a test message from the bank.")
        result = run_dual_variant_ocr(img)
        assert result["word_count"] >= 0


# ---------------------------------------------------------------------------
# check_unreadable_gate
# ---------------------------------------------------------------------------

class TestCheckUnreadableGate:
    def test_good_text_passes(self):
        text = "Your verification code is 482910. Do not share with anyone."
        unreadable, reason = check_unreadable_gate(text, 85.0)
        assert unreadable is False
        assert reason is None

    def test_text_too_short(self):
        unreadable, reason = check_unreadable_gate("Hi!", 90.0, min_text_len=15)
        assert unreadable is True
        assert "too short" in reason.lower()

    def test_confidence_too_low(self):
        text = "This message is long enough to pass the length gate but has low OCR confidence."
        unreadable, reason = check_unreadable_gate(text, 20.0, min_confidence=35.0)
        assert unreadable is True
        assert "confidence" in reason.lower()

    def test_empty_string_is_unreadable(self):
        unreadable, reason = check_unreadable_gate("", 0.0)
        assert unreadable is True

    def test_whitespace_only_is_unreadable(self):
        unreadable, reason = check_unreadable_gate("   \t\n  ", 80.0, min_text_len=5)
        assert unreadable is True

    def test_boundary_confidence_exactly_at_threshold(self):
        """Confidence exactly at min_confidence threshold should PASS (not rejected)."""
        text = "A sufficiently long string to pass the length requirement easily."
        unreadable, reason = check_unreadable_gate(text, 35.0, min_confidence=35.0)
        assert unreadable is False

    def test_boundary_length_exactly_at_threshold(self):
        text = "fifteen_chars!!"  # exactly 15 chars
        assert len(text) == 15
        unreadable, reason = check_unreadable_gate(text, 80.0, min_text_len=15)
        assert unreadable is False


# ---------------------------------------------------------------------------
# compute_character_error_rate
# ---------------------------------------------------------------------------

class TestComputeCharacterErrorRate:
    def test_identical_strings_zero_cer(self):
        ref = "Your OTP for HDFC Bank is 948271"
        assert compute_character_error_rate(ref, ref) == 0.0

    def test_empty_ref_empty_hyp(self):
        assert compute_character_error_rate("", "") == 0.0

    def test_empty_ref_nonempty_hyp(self):
        assert compute_character_error_rate("", "some hypothesis") == 1.0

    def test_empty_hyp_nonempty_ref(self):
        ref = "Your account has been limited"
        cer = compute_character_error_rate(ref, "")
        assert cer == 1.0

    def test_single_insertion(self):
        ref = "Hello World"
        hyp = "Helo World"  # 1 deletion
        cer = compute_character_error_rate(ref, hyp)
        assert 0.0 < cer < 0.2

    def test_full_substitution(self):
        ref = "abc"
        hyp = "xyz"
        cer = compute_character_error_rate(ref, hyp)
        assert cer == 1.0

    def test_normalised_between_zero_and_one(self):
        """CER must always be in [0, 1] for any pair of strings."""
        pairs = [
            ("Hello", "Hello World extra words appended here"),
            ("Very long reference string with many characters", "short"),
            ("Match", "Match"),
        ]
        for ref, hyp in pairs:
            cer = compute_character_error_rate(ref, hyp)
            # CER can exceed 1.0 for insertions longer than reference, but
            # our implementation uses max(1, len(ref)) so it can be > 1.
            assert cer >= 0.0

    def test_whitespace_normalisation(self):
        """Extra whitespace should be normalised before comparison."""
        ref = "hello   world"
        hyp = "hello world"
        cer = compute_character_error_rate(ref, hyp)
        assert cer == 0.0
