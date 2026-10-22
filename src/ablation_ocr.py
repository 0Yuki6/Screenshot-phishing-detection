"""
Validation set ablation: Dual-Variant vs Single CLAHE vs Single Adaptive Threshold.
"""

import os
import json
import cv2
import pandas as pd
from src.ocr import extract_text_and_confidence
from src.preprocess import resize_image_clamped, generate_preprocessing_variants


def run_ablation():
    df = pd.read_csv("data/processed/dataset_with_groups.csv")
    with open("data/processed/splits.json") as f:
        splits = json.load(f)

    # Sample 60 validation images balanced across channels
    val_indices = splits["val"]
    val_df = df.iloc[val_indices]

    # Sample 20 per channel
    sampled_indices = []
    for ch in ["email", "sms", "whatsapp"]:
        sub = val_df[val_df["channel"] == ch]
        sampled_indices.extend(sub.head(20).index.tolist())

    sample_val = df.loc[sampled_indices]

    results = {
        "clahe_conf": [], "adapt_conf": [], "dual_conf": [],
        "clahe_words": [], "adapt_words": [], "dual_words": []
    }

    for idx, r in sample_val.iterrows():
        p = os.path.join("data", "raw", r["channel"], r["label"], r["message_id"])
        if not os.path.exists(p):
            continue
        img = cv2.imread(p)
        if img is None:
            continue
        resized, _ = resize_image_clamped(img)
        va, vb = generate_preprocessing_variants(resized)

        ta, ca, wa = extract_text_and_confidence(va)
        tb, cb, wb = extract_text_and_confidence(vb)

        dual_c = max(ca, cb)
        dual_w = wa if ca >= cb else wb

        results["clahe_conf"].append(ca)
        results["adapt_conf"].append(cb)
        results["dual_conf"].append(dual_c)
        results["clahe_words"].append(wa)
        results["adapt_words"].append(wb)
        results["dual_words"].append(dual_w)

    res_df = pd.DataFrame(results)
    n = len(res_df)
    print("======================================================================")
    print(" OCR PREPROCESSING ABLATION STUDY (VALIDATION SET)")
    print("======================================================================")
    print(f"Evaluated on {n} validation samples (20 email, 20 sms, 20 whatsapp)")
    print(f"Single Adaptive Threshold: Mean Conf = {res_df['adapt_conf'].mean():.2f}% | Mean Words = {res_df['adapt_words'].mean():.1f}")
    print(f"Single CLAHE:              Mean Conf = {res_df['clahe_conf'].mean():.2f}% | Mean Words = {res_df['clahe_words'].mean():.1f}")
    print(f"Dual-Variant (Selected):   Mean Conf = {res_df['dual_conf'].mean():.2f}% | Mean Words = {res_df['dual_words'].mean():.1f}")

    clahe_wins = (res_df["clahe_conf"] > res_df["adapt_conf"]).sum()
    adapt_wins = (res_df["adapt_conf"] > res_df["clahe_conf"]).sum()
    ties = (res_df["clahe_conf"] == res_df["adapt_conf"]).sum()
    print(f"\nVariant Win Breakdown:")
    print(f"  CLAHE wins:            {clahe_wins} ({clahe_wins/n*100:.1f}%)")
    print(f"  Adaptive Thresh wins:  {adapt_wins} ({adapt_wins/n*100:.1f}%)")
    print(f"  Ties:                  {ties} ({ties/n*100:.1f}%)")
    print("======================================================================")
    return res_df


if __name__ == "__main__":
    run_ablation()
