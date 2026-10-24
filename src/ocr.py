"""
Dual-variant OCR execution wrapper with confidence scoring, unreadable gating,
and Character Error Rate (CER) measurement.
"""

import os
import re
from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import pytesseract
from pytesseract import Output
import Levenshtein

from src.preprocess import resize_image_clamped, generate_preprocessing_variants, crop_phone_system_bars

# Configure Tesseract binary path
_DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_DEFAULT_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = _DEFAULT_TESSERACT_PATH


def extract_text_and_confidence(img: np.ndarray, psm: int = 6) -> Tuple[str, float, int]:
    """
    Run Tesseract OCR on a preprocessed 2D/3D image array and compute mean word confidence.
    
    Args:
        img: Preprocessed grayscale or binary image array.
        psm: Page segmentation mode (default 6: assume a single uniform block of text).
        
    Returns:
        (extracted_text, mean_word_confidence, valid_word_count)
    """
    config = f"--psm {psm}"
    try:
        data = pytesseract.image_to_data(img, output_type=Output.DATAFRAME, config=config)
    except Exception as e:
        # Fallback to simple image_to_string if image_to_data errors
        text = pytesseract.image_to_string(img, config=config)
        return text.strip(), 0.0, len(text.split())

    # Filter out empty or whitespace-only words and invalid confidences (-1)
    valid_words = data[data["text"].notnull() & (data["conf"] >= 0)].copy()
    valid_words["text_clean"] = valid_words["text"].astype(str).str.strip()
    valid_words = valid_words[valid_words["text_clean"].str.len() > 0]

    if len(valid_words) == 0:
        return "", 0.0, 0

    mean_conf = float(valid_words["conf"].mean())
    words = valid_words["text_clean"].tolist()
    text = " ".join(words)
    return text.strip(), round(mean_conf, 2), len(words)


def extract_text(img: np.ndarray, psm: int = 6) -> str:
    """Convenience function returning extracted text."""
    text, _, _ = extract_text_and_confidence(img, psm=psm)
    return text


def run_dual_variant_ocr(
    bgr_img: np.ndarray,
    min_side: int = 600,
    max_side: int = 1920,
    psm: int = 6
) -> Dict[str, Any]:
    """
    Execute dual-variant OCR pipeline:
      1. Resize image preserving aspect ratio (600px - 1920px).
      2. Produce Variant A (CLAHE) and Variant B (Adaptive Gaussian Threshold).
      3. Run Tesseract on both variants.
      4. Select the variant with higher mean word confidence.
      
    Returns dictionary with:
        'text': selected text
        'confidence': selected mean word confidence (0-100)
        'word_count': number of words extracted
        'winning_variant': 'clahe' or 'adaptive_threshold'
        'variant_a_conf': confidence of CLAHE variant
        'variant_b_conf': confidence of adaptive threshold variant
    """
    # Geometrically crop status and navigation bars for phone-size images before OCR
    img_to_process, was_phone_cropped = crop_phone_system_bars(bgr_img)

    resized, scale = resize_image_clamped(img_to_process, min_side=min_side, max_side=max_side)
    var_a_clahe, var_b_adapt = generate_preprocessing_variants(resized)

    text_a, conf_a, words_a = extract_text_and_confidence(var_a_clahe, psm=psm)
    text_b, conf_b, words_b = extract_text_and_confidence(var_b_adapt, psm=psm)

    # Decision rule: pick higher confidence; if tied, pick higher word count
    if (conf_a > conf_b) or (abs(conf_a - conf_b) < 1e-2 and words_a >= words_b):
        winning_variant = "clahe"
        chosen_text = text_a
        chosen_conf = conf_a
        chosen_words = words_a
    else:
        winning_variant = "adaptive_threshold"
        chosen_text = text_b
        chosen_conf = conf_b
        chosen_words = words_b

    chosen_text = normalize_ocr_urls(chosen_text)

    return {
        "text": chosen_text,
        "confidence": chosen_conf,
        "word_count": chosen_words,
        "winning_variant": winning_variant,
        "variant_a_conf": conf_a,
        "variant_b_conf": conf_b,
        "scale_factor": scale,
        "phone_cropped": was_phone_cropped,
    }


def normalize_ocr_urls(text: str) -> str:
    """
    Rejoin URLs split across lines or by spaces during OCR extraction.
    E.g. 'http://' + line break/space + 'www...' -> 'http://www...'
    """
    if not text:
        return ""
    # Rejoin protocol + space/line break + domain or path
    t = re.sub(r'(https?:\s*//)\s+', r'\1', text, flags=re.IGNORECASE)
    t = re.sub(r'(https?://)\s+([a-zA-Z0-9_\-\./?=&%#]+)', r'\1\2', t, flags=re.IGNORECASE)
    # Rejoin line break after scheme or www
    t = re.sub(r'(https?://[^\s<>"\'\)\]]*?)[\r\n]+\s*([a-zA-Z0-9_\-\./?=&%#]+)', r'\1\2', t, flags=re.IGNORECASE)
    # Rejoin broken 'www.'
    t = re.sub(r'(www\.)\s+([a-zA-Z0-9_\-\./?=&%#]+)', r'\1\2', t, flags=re.IGNORECASE)
    return t


def check_unreadable_gate(
    text: str,
    confidence: float,
    min_text_len: int = 15,
    min_confidence: float = 35.0,
    min_alpha_ratio: float = 0.40,
    max_single_char_ratio: float = 0.30,
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate if OCR extraction is unreadable or out-of-distribution (garbled noise).

    Checks:
    1. Minimum text length (rejects empty/near-empty extracts).
    2. Mean word confidence (rejects severely degraded renders).
    3. Alphanumeric density (rejects symbol/artifact clusters).
    4. Fragment noise (rejects disconnected single-character OCR debris).

    Returns:
        (is_unreadable, reason_string_or_None)
    """
    clean = text.strip() if text else ""
    if len(clean) < min_text_len:
        return True, f"Extracted text too short ({len(clean)} chars < {min_text_len})"

    if confidence < min_confidence:
        return True, f"Mean OCR word confidence too low ({confidence:.1f}% < {min_confidence:.1f}%)"

    # Alphanumeric ratio check: garbled screenshots produce mostly punctuation / symbols
    clean_chars = [c for c in clean if c.isalnum() or c.isspace()]
    alpha_ratio = len(clean_chars) / max(1, len(clean))
    if alpha_ratio < min_alpha_ratio:
        return True, f"Garbled OCR: Low alphanumeric ratio ({alpha_ratio:.1%} < {min_alpha_ratio:.0%}) indicates visual noise"

    # Fragment noise check: garbled OCR produces broken single-letter tokens
    tokens = [w for w in clean.split() if w]
    if len(tokens) >= 8:
        single_chars = [w for w in tokens if len(w) == 1 and not w.isdigit() and w.lower() not in {"a", "i"}]
        single_ratio = len(single_chars) / len(tokens)
        if single_ratio > max_single_char_ratio:
            return True, f"Garbled OCR: Excessive fragment noise ({single_ratio:.1%} > {max_single_char_ratio:.0%} single-char tokens)"

    return False, None


def compute_character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate (CER) = Levenshtein_distance(ref, hyp) / len(ref).
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    ref_clean = " ".join(reference.split())
    hyp_clean = " ".join(hypothesis.split())
    dist = Levenshtein.distance(ref_clean, hyp_clean)
    return round(dist / max(1, len(ref_clean)), 4)
