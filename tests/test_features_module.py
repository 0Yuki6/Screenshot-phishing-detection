"""
Unit tests for the unified features module (src/features.py).
Verifies scalar feature extraction, split-safe TF-IDF vectorization,
and zero data leakage.
"""

import pytest
import numpy as np
import pandas as pd
from src.features import extract_features_single, build_feature_matrix, FEATURE_NAMES


def test_extract_features_single():
    text = "Dear Customer, please verify your bank account immediately at http://bit.ly/123"
    feats = extract_features_single(text, "email")
    assert isinstance(feats, dict)
    for name in FEATURE_NAMES:
        assert name in feats
    assert feats["channel"] == 0  # email
    assert feats["is_shortened_url"] == 1
    assert feats["urgency_count"] >= 1
    assert feats["domain_age"] == -1  # offline by default


def test_build_feature_matrix_training_fit():
    df_train = pd.DataFrame({
        "clean_text": [
            "Your HDFC account statement is ready.",
            "Immediate action required to avoid suspension at http://fake.com",
            "Meeting scheduled for tomorrow at 10am.",
            "OTP verification code is 492018.",
        ],
        "channel": ["email", "sms", "email", "sms"],
        "label": ["legitimate", "phishing", "legitimate", "legitimate"],
    })
    X, y, feat_names, vectorizer = build_feature_matrix(df_train, fit_tfidf=True, max_tfidf_features=50)

    assert X.shape[0] == 4
    assert y.shape[0] == 4
    assert np.array_equal(y, [0, 1, 0, 0])
    assert vectorizer is not None
    assert len(feat_names) == len(FEATURE_NAMES) + len(vectorizer.get_feature_names_out())
    assert X.shape[1] == len(feat_names)


def test_build_feature_matrix_val_no_leakage():
    # 1. Fit on train
    df_train = pd.DataFrame({
        "clean_text": [
            "HDFC Bank OTP is 1234.",
            "Suspicious login alert from new device.",
        ],
        "channel": ["sms", "email"],
        "label": ["legitimate", "phishing"],
    })
    X_train, y_train, feat_names_train, vectorizer = build_feature_matrix(
        df_train, fit_tfidf=True, max_tfidf_features=20
    )

    # 2. Transform val with pre-fitted vectorizer (fit_tfidf=False)
    df_val = pd.DataFrame({
        "clean_text": [
            "Your order has shipped. Track at amazon.com",
        ],
        "channel": ["email"],
        "label": ["legitimate"],
    })
    X_val, y_val, feat_names_val, _ = build_feature_matrix(
        df_val, tfidf_vectorizer=vectorizer, fit_tfidf=False
    )

    # Verify identical feature dimensions and feature names
    assert X_val.shape[0] == 1
    assert X_val.shape[1] == X_train.shape[1]
    assert feat_names_val == feat_names_train
    assert y_val[0] == 0
