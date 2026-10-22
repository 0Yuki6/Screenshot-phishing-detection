"""
Training script for real dataset images.

Traverses the data/raw/ directory, extracts text from images using the
preprocessing and OCR modules, builds the feature matrix, trains the model,
and saves the artifacts.

Usage:
    python -m src.train_on_real_data
"""

import os
import sys
import logging
import json
from glob import glob

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_image
from src.ocr import extract_text
from src.generate_synthetic_data import build_feature_matrix
from src.model import train_model, evaluate_model, save_model, get_permutation_importance

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = "data/raw"


def load_ocr_cache() -> pd.DataFrame:
    cache_path = "data/processed/ocr_cache.csv"
    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    return pd.DataFrame(columns=["message_id", "ocr_text", "channel", "label"])


def save_ocr_cache(df: pd.DataFrame):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/ocr_cache.csv", index=False)


def _process_single_image(args):
    filepath, filename, channel, label = args
    logger.info("Processing %s", filepath)
    try:
        with open(filepath, "rb") as f:
            image_bytes = f.read()
            
        binarized_image = preprocess_image(image_bytes)
        text = extract_text(binarized_image)
        
        if not text.strip():
            logger.warning("Empty OCR text for %s. Skipping.", filepath)
            return None
            
        return {
            "message_id": filename,
            "ocr_text": text,
            "channel": channel,
            "label": label,
        }
    except Exception as e:
        logger.error("Failed to process %s: %s", filepath, e)
        return None


def extract_data_from_images() -> pd.DataFrame:
    """
    Traverse the raw data directory and process each image (with caching and multiprocessing).
    
    Returns:
        DataFrame with columns: message_id, ocr_text, channel, label
    """
    supported_exts = {".png", ".jpg", ".jpeg", ".webp"}
    
    cache_df = load_ocr_cache()
    cached_ids = set(cache_df["message_id"].values)
    
    tasks = []
    if os.path.exists(RAW_DATA_DIR):
        channels = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
        for channel in channels:
            for label in ["legitimate", "phishing"]:
                label_dir = os.path.join(RAW_DATA_DIR, channel, label)
                if not os.path.exists(label_dir):
                    continue
                    
                for filename in os.listdir(label_dir):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in supported_exts:
                        continue
                        
                    if filename in cached_ids:
                        continue
                        
                    filepath = os.path.join(label_dir, filename)
                    tasks.append((filepath, filename, channel, label))

    new_records = []
    if tasks:
        import concurrent.futures
        logger.info(f"Running OCR on {len(tasks)} new images using multiprocessing...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(_process_single_image, tasks):
                if result:
                    new_records.append(result)
                    
    if new_records:
        new_df = pd.DataFrame(new_records)
        # Exclude empty or all-NA entries before concat if they exist, though new_df is clean
        cache_df = pd.concat([cache_df, new_df], ignore_index=True)
        save_ocr_cache(cache_df)

    logger.info("Extracted data from %d real images", len(cache_df))
    return cache_df


def main():
    """Main training pipeline."""
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)

    # Step 1: Extract data
    logger.info("=== Step 1: Extracting text from real images ===")
    df = extract_data_from_images()
    
    if df.empty:
        logger.error("No valid data found in %s. Exiting.", RAW_DATA_DIR)
        return
        
    df.to_csv("data/processed/real_samples.csv", index=False)
    logger.info("Saved extracted text to data/processed/real_samples.csv")

    # Step 2: Build feature matrix
    logger.info("=== Step 2: Extracting features ===")
    X, y, feature_names, tfidf = build_feature_matrix(df)

    feature_df = pd.DataFrame(X, columns=feature_names)
    feature_df["label"] = y
    feature_df["channel_name"] = df["channel"].values
    feature_df.to_csv("data/processed/features_real.csv", index=False)
    logger.info("Saved feature matrix to data/processed/features_real.csv")

    # Step 3: Train/test split and train model
    logger.info("=== Step 3: Training model ===")
    # Handle very small datasets during testing gracefully
    test_size = 0.2 if len(df) >= 10 else 0.5
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 and len(df) >= 10 else None
    )
    model = train_model(X_train, y_train)

    # Step 4: Evaluate
    logger.info("=== Step 4: Evaluating model ===")
    try:
        results = evaluate_model(model, X_test, y_test, label_names=["legitimate", "phishing"])
        print("\n" + results["classification_report"])
        print(f"Confusion Matrix:\n{results['confusion_matrix']}\n")
    except Exception as e:
        logger.warning("Evaluation failed (likely too few samples): %s", e)

    # Step 5: Save model and TF-IDF vectorizer
    logger.info("=== Step 5: Saving artifacts ===")
    save_model(model, "data/models/model.pkl")
    joblib.dump(tfidf, "data/models/tfidf_vectorizer.pkl")
    logger.info("Saved TF-IDF vectorizer to data/models/tfidf_vectorizer.pkl")

    with open("data/models/feature_names.json", "w") as f:
        json.dump(feature_names, f)
    logger.info("Saved feature names to data/models/feature_names.json")

    # Step 6: Permutation importance
    logger.info("=== Step 6: Feature importance ===")
    try:
        importance = get_permutation_importance(model, X_test, y_test, feature_names)
        print("\nTop 15 Features by Permutation Importance:")
        for name, mean, std in importance[:15]:
            print(f"  {name:40s} {mean:+.4f} ± {std:.4f}")
    except Exception as e:
        logger.warning("Feature importance failed: %s", e)

    logger.info("\n✓ Pipeline complete. Artifacts saved to data/models/")


if __name__ == "__main__":
    main()
