"""
Diagnostic script for the phishing detection pipeline.

For each image path given as CLI arguments (or auto-discovered from data/raw),
prints:
  - OCR text (first 400 chars)
  - Detected channel
  - Raw predict_proba (both classes)
  - Feature vector shape vs model.n_features_in_
  - Whether feature_names.json matches FEATURE_NAMES + tfidf names in order
  - Values of: brand_domain_mismatch, has_high_risk_tld, blacklist_status, whitelist_match

Usage:
    python debug_one.py [image_path1] [image_path2] ...
    python debug_one.py  # auto-discovers from data/raw (3 per class per channel)
"""

import sys
import os
import json
import glob

import cv2
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ocr import run_dual_variant_ocr
from src.channel_detector import detect_channel
from src.masking import strip_template_chrome, mask_text
from src.feature_extraction import extract_features, FEATURE_NAMES

MODEL_PATH = "data/models/calibrated_model.pkl"
TFIDF_PATH = "data/models/tfidf_vectorizer.pkl"
FEATURE_NAMES_PATH = "data/models/feature_names.json"


def auto_discover_images(base_dir="data/raw", per_class=2):
    """Find sample images from data/raw, per_class images per class per channel."""
    images = []
    for channel in ["email", "sms", "whatsapp"]:
        for label in ["legitimate", "phishing"]:
            pattern = os.path.join(base_dir, channel, label, "*")
            found = sorted(glob.glob(pattern))
            images.extend(found[:per_class])
    return images


def main():
    # Load model artifacts
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH) if os.path.exists(TFIDF_PATH) else None

    # Load feature_names.json
    feature_names_json = None
    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH) as f:
            feature_names_json = json.load(f)

    # Determine image paths
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
    else:
        image_paths = auto_discover_images()
        if not image_paths:
            print("No images found. Pass image paths as arguments.")
            return

    # Build expected feature names (FEATURE_NAMES + tfidf names)
    if tfidf is not None:
        tfidf_names = [f"tfidf_{name}" for name in tfidf.get_feature_names_out()]
        expected_names = list(FEATURE_NAMES) + tfidf_names
    else:
        expected_names = list(FEATURE_NAMES)

    # Check alignment
    names_match = feature_names_json == expected_names if feature_names_json else False

    print("=" * 80)
    print("FEATURE NAME ALIGNMENT CHECK")
    print(f"  FEATURE_NAMES count: {len(FEATURE_NAMES)}")
    print(f"  TF-IDF features: {len(tfidf_names) if tfidf else 0}")
    print(f"  Expected total: {len(expected_names)}")
    print(f"  feature_names.json count: {len(feature_names_json) if feature_names_json else 'N/A'}")
    print(f"  model.n_features_in_: {getattr(model, 'n_features_in_', 'N/A')}")
    print(f"  Names match exactly: {names_match}")

    if not names_match and feature_names_json:
        # Find mismatches
        for i, (exp, got) in enumerate(zip(expected_names, feature_names_json)):
            if exp != got:
                print(f"  FIRST MISMATCH at index {i}: expected='{exp}', got='{got}'")
                break
        if len(expected_names) != len(feature_names_json):
            print(f"  LENGTH MISMATCH: expected {len(expected_names)}, got {len(feature_names_json)}")

    print("=" * 80)
    print()

    for img_path in image_paths:
        print("-" * 80)
        print(f"IMAGE: {img_path}")

        # Derive true label from path
        path_lower = img_path.replace("\\", "/").lower()
        if "/phishing/" in path_lower:
            true_label = "phishing"
        elif "/legitimate/" in path_lower:
            true_label = "legitimate"
        else:
            true_label = "unknown"
        print(f"  True label (from path): {true_label}")

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ERROR: Could not read image")
            continue

        # OCR
        ocr_result = run_dual_variant_ocr(img)
        raw_text = ocr_result["text"]
        ocr_conf = ocr_result["confidence"]

        print(f"  OCR confidence: {ocr_conf}")
        print(f"  OCR text (first 400 chars):")
        print(f"    {repr(raw_text[:400])}")
        print()

        # Channel
        ch_info = detect_channel(raw_text)
        channel = ch_info.get("channel", "email")
        print(f"  Detected channel: {channel}")

        # Clean text
        clean_text = strip_template_chrome(raw_text, channel=channel)

        # Extract features
        feat_dict = extract_features(clean_text, channel, allow_network=False, skip_whois=True)
        scalar_vector = np.array([[feat_dict[name] for name in FEATURE_NAMES]], dtype=np.float64)

        # TF-IDF
        if tfidf is not None:
            masked_text = mask_text(clean_text)
            tfidf_vec = tfidf.transform([masked_text])
            X = hstack([csr_matrix(scalar_vector), tfidf_vec]).toarray()
        else:
            X = scalar_vector

        # Shape check
        n_features_in = getattr(model, "n_features_in_", None)
        print(f"  Feature vector shape: {X.shape}")
        print(f"  Model expects: {n_features_in}")
        print(f"  Shape match: {X.shape[1] == n_features_in if n_features_in else 'N/A'}")

        # Predict
        try:
            proba = model.predict_proba(X)[0]
            classes = model.classes_
            print(f"  model.classes_: {classes}")
            print(f"  predict_proba: {dict(zip(classes, proba))}")

            # p_phish = proba for class 1
            phish_idx = list(classes).index(1)
            p_phish = proba[phish_idx]
            print(f"  p_phish (class 1): {p_phish:.4f}")
        except Exception as e:
            print(f"  PREDICTION ERROR: {e}")
            p_phish = None

        # Key feature values
        print(f"  KEY FEATURES:")
        for key in ["brand_domain_mismatch", "has_high_risk_tld", "blacklist_status", "whitelist_match",
                     "urgency_count", "threat_count", "credential_request", "sensitive_request",
                     "typosquatting_score", "url_length"]:
            val = feat_dict.get(key, "MISSING")
            print(f"    {key}: {val}")

        print()


if __name__ == "__main__":
    main()
