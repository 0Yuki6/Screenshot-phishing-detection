"""
Unit tests for dataset preparation, template extraction, and leakage-free group splitting.
"""

import os
import json
import pandas as pd
import pytest

from src.dataset import extract_template_label, compute_group_ids, create_grouped_splits


def test_extract_template_label():
    assert extract_template_label("legitimate_0002_outlook.png", "email") == "outlook"
    assert extract_template_label("phishing_0005_gmail.png", "email") == "gmail"
    assert extract_template_label("legit_0010_yahoo.png", "email") == "yahoo"
    assert extract_template_label("phish_0020_applemail.png", "email") == "applemail"
    assert extract_template_label("legit_0001.png", "sms") == "standard_sms"
    assert extract_template_label("phishing_0001.png", "whatsapp") == "standard_whatsapp"


def test_compute_group_ids_clustering():
    texts = [
        "Your PayPal account has been restricted. Verify at http://evil.xyz",
        "Your PayPal account has been restricted. Verify at http://evil.xyz",  # identical
        "Your PayPal account has been restricted. Verify at http://evil.xyz?token=123",  # near duplicate
        "Meeting reminder for tomorrow 10am in conference room B",  # completely different
        "Meeting reminder for tomorrow 10am in conference room B with client",  # near duplicate
    ]
    groups = compute_group_ids(texts, similarity_threshold=0.80)
    assert len(groups) == 5
    # First 3 should belong to the same group
    assert groups[0] == groups[1] == groups[2]
    # Last 2 should belong to another group
    assert groups[3] == groups[4]
    # First cluster and second cluster must be distinct
    assert groups[0] != groups[3]


def test_zero_leakage_in_splits():
    # Synthetic dataframe with groups
    data = []
    for g in range(100):
        # 1 to 4 samples per group
        for s in range((g % 4) + 1):
            data.append({
                "group_id": g,
                "channel": "email" if g < 50 else "sms",
                "label": "phishing" if (g % 2 == 0) else "legitimate",
            })
    df = pd.DataFrame(data)

    splits = create_grouped_splits(df, seed=42, test_ratio=0.2, val_ratio=0.2)
    train_groups = set(df.iloc[splits["train"]]["group_id"])
    val_groups = set(df.iloc[splits["val"]]["group_id"])
    test_groups = set(df.iloc[splits["test"]]["group_id"])

    assert len(train_groups.intersection(val_groups)) == 0
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(val_groups.intersection(test_groups)) == 0


def test_dataset_processed_artifacts_exist():
    assert os.path.exists("data/processed/dataset_with_groups.csv")
    assert os.path.exists("data/processed/splits.json")
    with open("data/processed/splits.json") as f:
        splits = json.load(f)
    assert "train" in splits and "val" in splits and "test" in splits
    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0
