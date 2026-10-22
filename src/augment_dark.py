"""
Generate dark-mode augmented screenshots from main-dataset training templates.

Applies high-fidelity LAB luminance inversion to preserve hue/chroma while inverting
light backgrounds to dark mode.
Assigns each dark image to the exact same `group_id` as its light original and places
it strictly in the `train` split (`source='augmented_dark'`).
"""

import os
import concurrent.futures
from typing import Dict, Any, List

import cv2
import pandas as pd
from src.ocr import run_dual_variant_ocr


def convert_to_dark_mode(bgr_img):
    """
    Invert luminance in LAB space to create realistic dark mode screens
    while preserving chroma and accent colors.
    """
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_inv = cv2.bitwise_not(l)
    return cv2.cvtColor(cv2.merge([l_inv, a, b]), cv2.COLOR_LAB2BGR)


def process_dark_sample(args) -> Dict[str, Any]:
    row, out_dir = args
    in_path = os.path.join("data", "raw", row["channel"], row["label"], row["message_id"])
    if not os.path.exists(in_path):
        return None

    img = cv2.imread(in_path)
    if img is None:
        return None

    dark_img = convert_to_dark_mode(img)
    out_sub = os.path.join(out_dir, row["channel"], row["label"])
    os.makedirs(out_sub, exist_ok=True)
    out_filename = f"dark_{row['message_id']}"
    out_path = os.path.join(out_sub, out_filename)
    cv2.imwrite(out_path, dark_img)

    # Run dual OCR on the dark image
    res = run_dual_variant_ocr(dark_img)

    return {
        "message_id": out_filename,
        "channel": row["channel"],
        "label": row["label"],
        "template": row["template"],
        "ocr_text": res["text"],
        "confidence": res["confidence"],
        "winning_variant": res["winning_variant"],
        "group_id": row["group_id"],  # Grouped strictly with light original
        "split": "train",             # Kept strictly in training split
        "source": "augmented_dark",
    }


def generate_dark_augmented_dataset(
    dataset_csv: str = "data/processed/dataset_with_groups.csv",
    output_dir: str = "data/augmented_dark",
    samples_per_template: int = 50,  # 50 legit + 50 phish per template = 600 total
    seed: int = 42,
    max_workers: int = 8,
):
    df = pd.read_csv(dataset_csv)
    train_df = df[df["split"] == "train"].copy()

    # Sample 50 legit and 50 phish per template
    selected_rows = []
    for (channel, template, label), group in train_df.groupby(["channel", "template", "label"]):
        sampled = group.sample(n=min(len(group), samples_per_template), random_state=seed)
        selected_rows.extend(sampled.to_dict("records"))

    print(f"Generating and OCR'ing {len(selected_rows)} dark-mode augmented screenshots across {train_df['template'].nunique()} templates...")

    tasks = [(row, output_dir) for row in selected_rows]
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(process_dark_sample, tasks):
            if res:
                records.append(res)
            if len(records) % 100 == 0 and len(records) > 0:
                print(f"  Processed {len(records)}/{len(tasks)} dark-augmented images...")

    df_dark = pd.DataFrame(records)
    out_csv = "data/processed/augmented_dark_samples.csv"
    df_dark.to_csv(out_csv, index=False)
    print(f"Saved {len(df_dark)} dark augmented records to {out_csv}")
    print(f"Mean OCR confidence on dark augmented images: {df_dark['confidence'].mean():.2f}%")
    print(f"Winning variant breakdown:\n{df_dark['winning_variant'].value_counts()}")
    return df_dark


if __name__ == "__main__":
    generate_dark_augmented_dataset()
