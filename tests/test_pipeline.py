"""
Unit tests for the end-to-end PhishingPipeline (src/pipeline.py).
"""

import os
import pytest
import numpy as np
import cv2

from src.pipeline import PhishingPipeline


@pytest.fixture(scope="module")
def pipeline():
    return PhishingPipeline()


def test_pipeline_blank_image_unreadable(pipeline):
    # Blank 100x100 black image
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    res = pipeline.predict_image(blank)

    assert isinstance(res, dict)
    assert res["verdict"] == "uncertain"
    assert "UNREADABLE_OCR" in res["rules_triggered"]
    assert res["ocr_confidence"] == 0.0 or len(res["clean_text"]) < 15


def test_pipeline_schema_keys(pipeline):
    # Generate test image with some text
    img = np.full((300, 800, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Your HDFC Bank account statement is ready.", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Visit https://netbanking.hdfcbank.com/entry", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    res = pipeline.predict_image(img, channel_hint="email")

    required_keys = [
        "verdict",
        "phishing_score",
        "channel",
        "alerts",
        "extracted_text",
        "clean_text",
        "ocr_confidence",
        "winning_variant",
        "rules_triggered",
        "raw_ml_score",
    ]
    for key in required_keys:
        assert key in res, f"Missing key: {key}"

    assert res["verdict"] in {"legitimate", "phishing", "uncertain"}
    assert 0.0 <= res["phishing_score"] <= 1.0
    assert res["channel"] == "email"
    assert "EXACT_WHITELIST_BYPASS" in res["rules_triggered"]
    assert res["verdict"] == "legitimate"


def test_pipeline_phishing_escalation(pipeline):
    img = np.full((300, 800, 3), 255, dtype=np.uint8)
    cv2.putText(img, "PayPal: Your account is suspended. Verify now.", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Visit http://paypa1-update-account.com/login", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    res = pipeline.predict_image(img, channel_hint="sms")

    assert res["verdict"] == "phishing"
    assert res["phishing_score"] >= 0.90
    assert "BRAND_IMPERSONATION_MISMATCH" in res["rules_triggered"]
