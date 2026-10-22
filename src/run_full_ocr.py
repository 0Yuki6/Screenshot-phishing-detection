"""
Full OCR runner with multithreading and caching for the 6,000 main-dataset images.
"""

import os
import time
import json
import concurrent.futures
from typing import Dict, Any, Optional

import cv2
import pandas as pd
from src.ocr import run_dual_variant_ocr


def process_single_image(args) -> Optional[Dict[str, Any]]:
    row_idx, channel, label, message_id = args
    img_path = os.path.join("data", "raw", channel, label, message_id)
    if not os.path.exists(img_path):
        return None

    img = cv2.imread(img_path)
    if img is None:
        return None

    res = run_dual_variant_ocr(img)
    return {
        "message_id": message_id,
        "channel": channel,
        "label": label,
        "ocr_text_dual": res["text"],
        "confidence": res["confidence"],
        "word_count": res["word_count"],
        "winning_variant": res["winning_variant"],
    }


def run_full_ocr(
    input_csv: str = "data/processed/dataset_with_groups.csv",
    output_cache: str = "data/processed/dual_ocr_cache.csv",
    max_workers: int = 8,
    limit: Optional[int] = None,
):
    df = pd.read_csv(input_csv)
    if limit:
        df = df.head(limit)

    # Check existing cache
    cached_ids = set()
    records = []
    if os.path.exists(output_cache):
        try:
            cached_df = pd.read_csv(output_cache)
            cached_ids = set(cached_df["message_id"].tolist())
            records = cached_df.to_dict("records")
            print(f"Loaded {len(records)} existing cached OCR records.")
        except Exception as e:
            print(f"Cache load error: {e}")

    tasks = [
        (idx, row["channel"], row["label"], row["message_id"])
        for idx, row in df.iterrows()
        if row["message_id"] not in cached_ids
    ]

    print(f"Running dual-variant OCR on {len(tasks)} images with {max_workers} workers...")
    if tasks:
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            completed = 0
            for res in executor.map(process_single_image, tasks):
                if res:
                    records.append(res)
                    completed += 1
                if completed % 500 == 0 and completed > 0:
                    print(f"  Completed {completed}/{len(tasks)} images ({time.time() - t0:.1f}s)...")
                    pd.DataFrame(records).to_csv(output_cache, index=False)

        print(f"Finished processing in {time.time() - t0:.1f}s.")
        pd.DataFrame(records).to_csv(output_cache, index=False)
        print(f"Saved complete OCR cache to {output_cache}")

    return pd.DataFrame(records)


if __name__ == "__main__":
    run_full_ocr()
