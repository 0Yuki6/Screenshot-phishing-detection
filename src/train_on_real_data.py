"""
Ingestion and processing script for real training screenshots.

Scans data/real_train/<channel>/<label>/, validates against dev and final holdout
image hashes to prevent any data leakage, runs dual-variant OCR, and appends
processed samples to data/processed/real_train_samples.csv with source="real".
"""

import os
import sys
import hashlib
import glob
import logging
from typing import Set, Dict, List, Any

import cv2
import pandas as pd

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ocr import run_dual_variant_ocr
from src.masking import strip_template_chrome

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REAL_TRAIN_DIR = "data/real_train"
PROCESSED_REAL_TRAIN_CSV = "data/processed/real_train_samples.csv"
HOLDOUT_DIRS = [
    "data/real_holdout/dev",
    "data/real_holdout/final",
    "data/real_holdout_originals",
]


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file's raw bytes."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_holdout_hashes(holdout_dirs: List[str] = HOLDOUT_DIRS) -> Set[str]:
    """Collect SHA-256 hashes of all holdout images to guarantee zero leakage."""
    holdout_hashes: Set[str] = set()
    supported_exts = {".png", ".jpg", ".jpeg", ".webp"}

    for h_dir in holdout_dirs:
        if not os.path.exists(h_dir):
            continue
        for root, _, files in os.walk(h_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_exts:
                    fp = os.path.join(root, file)
                    holdout_hashes.add(compute_file_hash(fp))

    logger.info("Indexed %d unique holdout hashes across holdout sets.", len(holdout_hashes))
    return holdout_hashes


def ingest_real_training_data(
    real_train_dir: str = REAL_TRAIN_DIR,
    output_csv: str = PROCESSED_REAL_TRAIN_CSV,
) -> pd.DataFrame:
    """
    Process images from data/real_train/<channel>/<label>/, verify zero overlap
    with holdout data, run dual OCR, and output processed DataFrame.
    """
    channels = ["email", "sms", "whatsapp"]
    labels = ["legitimate", "phishing"]
    supported_exts = {".png", ".jpg", ".jpeg", ".webp"}

    # Create folder structure if it doesn't exist yet
    for ch in channels:
        for lbl in labels:
            os.makedirs(os.path.join(real_train_dir, ch, lbl), exist_ok=True)

    holdout_hashes = get_holdout_hashes()

    existing_df = pd.DataFrame()
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            logger.info("Loaded %d previously processed real training samples from %s", len(existing_df), output_csv)
        except Exception as e:
            logger.warning("Could not load existing %s: %s", output_csv, e)

    processed_ids = set(existing_df["message_id"].values) if not existing_df.empty else set()

    new_records: List[Dict[str, Any]] = []
    skipped_leakage_count = 0

    for ch in channels:
        for lbl in labels:
            folder = os.path.join(real_train_dir, ch, lbl)
            files = sorted(os.listdir(folder))
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in supported_exts:
                    continue

                fp = os.path.join(folder, fn)
                img_hash = compute_file_hash(fp)

                # Leakage check: NEVER allow holdout images into training
                if img_hash in holdout_hashes:
                    logger.warning("SKIPPING %s: Duplicate/Hash match with holdout evaluation set!", fp)
                    skipped_leakage_count += 1
                    continue

                msg_id = f"real_{ch}_{lbl}_{fn}"
                if msg_id in processed_ids:
                    continue

                img = cv2.imread(fp)
                if img is None:
                    logger.warning("Failed to read image %s. Skipping.", fp)
                    continue

                ocr_res = run_dual_variant_ocr(img)
                raw_text = ocr_res["text"]
                conf = float(ocr_res["confidence"])
                winner = ocr_res["winning_variant"]
                clean_t = strip_template_chrome(raw_text, channel=ch)

                new_records.append({
                    "message_id": msg_id,
                    "channel": ch,
                    "label": lbl,
                    "ocr_text": raw_text,
                    "clean_text": clean_t,
                    "confidence": conf,
                    "winning_variant": winner,
                    "source": "real",
                    "file_hash": img_hash,
                })
                logger.info("Ingested [%s/%s] %s (Conf: %.1f%%)", ch, lbl, fn, conf)

    if skipped_leakage_count > 0:
        logger.warning("Blocked %d images matching holdout dataset hashes.", skipped_leakage_count)

    if new_records:
        new_df = pd.DataFrame(new_records)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        combined_df.to_csv(output_csv, index=False)
        logger.info("Saved %d total real training samples to %s", len(combined_df), output_csv)
        return combined_df

    logger.info("No new real training images to ingest in %s.", real_train_dir)
    return existing_df


if __name__ == "__main__":
    ingest_real_training_data()
