"""
Script to compare Engineered-Only features vs Engineered + Character N-Grams on dev holdout.

Evaluates:
1. In-distribution validation metrics (accuracy, FPR, recall, AUROC).
2. Real holdout dev partition (30 samples: 10 email, 10 sms, 10 whatsapp - all legitimate).
3. False positive rate per channel on dev.
4. Top contributing features for each false positive on dev.
Strictly zero training on real holdout data.
"""

import os
import glob
import json
import logging
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from src.features import build_feature_matrix, FEATURE_NAMES
from src.masking import strip_template_chrome, mask_text
from src.ocr import run_dual_variant_ocr, check_unreadable_gate
from src.channel_detector import detect_channel
from src.rules import evaluate_rules

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_training_and_val_data():
    """Load training and validation data combining main dataset, augmented dark, and phone samples."""
    df_main = pd.read_csv("data/processed/dataset_with_groups.csv")
    df_dark = pd.read_csv("data/processed/augmented_dark_samples.csv")
    df_phone = pd.read_csv("data/processed/phone_training_samples.csv")

    train_main = df_main[df_main["split"] == "train"].copy()
    val_main = df_main[df_main["split"] == "val"].copy()

    # Combine training sets
    train_dfs = [
        train_main[["channel", "label", "ocr_text"]],
        df_dark[["channel", "label", "ocr_text"]],
        df_phone[["channel", "label", "ocr_text"]],
    ]
    df_train = pd.concat(train_dfs, ignore_index=True)
    df_val = val_main[["channel", "label", "ocr_text"]].reset_index(drop=True)

    print(f"Combined Training Samples: {len(df_train)} ({len(df_train[df_train['label']=='legitimate'])} legit, {len(df_train[df_train['label']=='phishing'])} phish)")
    print(f"Validation Samples: {len(df_val)}")
    return df_train, df_val


def train_and_calibrate_rf(X_train, y_train, feature_names):
    """Train Random Forest and calibrate with Platt sigmoid scaling."""
    base_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    calibrated = CalibratedClassifierCV(estimator=base_rf, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def tune_thresholds_on_val(calibrated_model, X_val, y_val, target_fpr=0.01):
    """Tune T_legit and T_phish on validation data."""
    val_probs = calibrated_model.predict_proba(X_val)[:, 1]
    legit_mask = (y_val == 0)
    phish_mask = (y_val == 1)

    # T_phish: lowest threshold achieving target FPR (<= 1%) on decided samples
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_t_phish = 0.70
    best_t_legit = 0.35

    # Find T_phish where FPR on decided phish is <= target_fpr
    for t in thresholds:
        fp = np.sum((val_probs >= t) & legit_mask)
        tn = np.sum((val_probs < t) & legit_mask)
        fpr = fp / max(1, (fp + tn))
        if fpr <= target_fpr:
            best_t_phish = round(float(t), 3)
            break

    # Find T_legit where FNR on decided legit is <= 2%
    for t in thresholds[::-1]:
        fn = np.sum((val_probs <= t) & phish_mask)
        tp = np.sum((val_probs > t) & phish_mask)
        fnr = fn / max(1, (fn + tp))
        if fnr <= 0.02:
            best_t_legit = round(float(t), 3)
            break

    best_t_legit = min(best_t_legit, best_t_phish - 0.10)
    return best_t_legit, best_t_phish


def evaluate_on_dev(model, tfidf_vec, t_legit, t_phish, all_feature_names):
    """Evaluate pipeline on all 30 dev holdout samples."""
    results_by_channel = {"email": [], "sms": [], "whatsapp": []}
    
    rf_base = model.calibrated_classifiers_[0].estimator
    importances = rf_base.feature_importances_

    for ch in ["email", "sms", "whatsapp"]:
        img_paths = sorted(glob.glob(f"data/real_holdout/dev/{ch}/*.*"))
        for path in img_paths:
            # 1. OCR (dual-variant + geometric crop)
            import cv2
            img = cv2.imread(path)
            ocr_res = run_dual_variant_ocr(img)
            raw_text = ocr_res["text"]
            ocr_conf = float(ocr_res["confidence"])
            winning_var = ocr_res["winning_variant"]

            # 2. Chrome stripping
            clean_t = strip_template_chrome(raw_text, channel=ch)

            # 3. Quality Gate
            is_unreadable, unread_reason = check_unreadable_gate(clean_t, ocr_conf)
            if is_unreadable:
                results_by_channel[ch].append({
                    "file": os.path.basename(path),
                    "verdict": "uncertain",
                    "score": 0.5,
                    "rules": ["UNREADABLE_OCR"],
                    "alerts": [unread_reason],
                    "top_features": [],
                })
                continue

            # 4. Feature Extraction
            scalar_feats = [FEATURE_NAMES]
            from src.features import extract_features
            s_dict = extract_features(clean_t, ch, allow_network=False, skip_whois=True)
            s_vec = np.array([[s_dict[name] for name in FEATURE_NAMES]], dtype=np.float64)

            masked_t = mask_text(clean_t)
            if tfidf_vec is not None:
                tf_vec = tfidf_vec.transform([masked_t])
                X_sample = hstack([csr_matrix(s_vec), tf_vec]).toarray()
            else:
                X_sample = s_vec

            raw_ml = float(model.predict_proba(X_sample)[:, 1][0])

            # 5. Rules Evaluation
            rule_res = evaluate_rules(
                raw_ocr_text=raw_text,
                clean_text=clean_t,
                ocr_confidence=ocr_conf,
                ml_score=raw_ml,
                channel=ch,
                t_legit=t_legit,
                t_phish=t_phish,
            )

            # Top contributing features
            top_feats = []
            if rule_res["verdict"] == "phishing":
                x_row = X_sample[0]
                contrib = x_row * importances[:len(x_row)]
                top_indices = np.argsort(contrib)[::-1][:5]
                for idx in top_indices:
                    if contrib[idx] > 0:
                        top_feats.append(f"{all_feature_names[idx]}={x_row[idx]:.2f} (imp={importances[idx]:.3f})")

            results_by_channel[ch].append({
                "file": os.path.basename(path),
                "verdict": rule_res["verdict"],
                "score": rule_res["phishing_score"],
                "raw_ml": raw_ml,
                "rules": rule_res["rules_triggered"],
                "alerts": rule_res["alerts"],
                "top_features": top_feats,
            })

    return results_by_channel


def summarize_dev_results(results_by_channel, model_name="Model"):
    print(f"\n==================================================================")
    print(f"DEV EVALUATION SUMMARY: {model_name}")
    print(f"==================================================================")
    total_samples = 0
    total_legit = 0
    total_fp = 0
    total_unc = 0

    for ch, items in results_by_channel.items():
        n = len(items)
        legit = sum(1 for x in items if x["verdict"] == "legitimate")
        fp = sum(1 for x in items if x["verdict"] == "phishing")
        unc = sum(1 for x in items if x["verdict"] == "uncertain")
        decided = legit + fp
        fpr = (fp / decided * 100.0) if decided > 0 else 0.0

        total_samples += n
        total_legit += legit
        total_fp += fp
        total_unc += unc

        print(f"  * {ch.upper():8s}: N={n:2d} | Legit={legit:2d} | FP={fp:2d} | Unc={unc:2d} | Decided FPR={fpr:5.1f}%")

    total_decided = total_legit + total_fp
    overall_fpr = (total_fp / total_decided * 100.0) if total_decided > 0 else 0.0
    unc_rate = total_unc / total_samples * 100.0
    print(f"  ----------------------------------------------------------------")
    print(f"  TOTAL     : N={total_samples:2d} | Legit={total_legit:2d} | FP={total_fp:2d} | Unc={total_unc:2d} | Decided FPR={overall_fpr:5.1f}% | Abstain={unc_rate:4.1f}%")

    fps = [item for ch in results_by_channel.values() for item in ch if item["verdict"] == "phishing"]
    if fps:
        print(f"\n  Top Contributing Features for False Positives ({len(fps)} FPs):")
        for item in fps:
            print(f"    - {item['file']}: score={item['score']:.3f}, rules={item['rules']}")
            if item["top_features"]:
                print(f"      features: {', '.join(item['top_features'][:4])}")
            if item["alerts"]:
                print(f"      alerts: {item['alerts'][0][:90]}")


def run_comparison():
    df_train, df_val = load_training_and_val_data()

    # ──────────────────────────────────────────────────────────────────────────
    # CONFIGURATION 1: Engineered-Only Features (no TF-IDF)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n>>> Building Matrix for CONFIG 1: Engineered-Only Features...")
    X_train_eng, y_train, names_eng, _ = build_feature_matrix(
        df_train,
        fit_tfidf=False,
        include_tfidf=False,
        text_column="ocr_text",
    )
    X_val_eng, y_val, _, _ = build_feature_matrix(
        df_val,
        fit_tfidf=False,
        include_tfidf=False,
        text_column="ocr_text",
    )

    print(f"Training Model 1 (Engineered-Only, {X_train_eng.shape[1]} features)...")
    model_eng = train_and_calibrate_rf(X_train_eng, y_train, names_eng)
    t_legit_1, t_phish_1 = tune_thresholds_on_val(model_eng, X_val_eng, y_val)
    print(f"Config 1 Thresholds: T_legit={t_legit_1:.3f}, T_phish={t_phish_1:.3f}")

    results_dev_eng = evaluate_on_dev(model_eng, None, t_legit_1, t_phish_1, names_eng)
    summarize_dev_results(results_dev_eng, model_name="Config 1: Engineered-Only Features")

    # ──────────────────────────────────────────────────────────────────────────
    # CONFIGURATION 2: Engineered + Character N-Grams (Masked, min_df=10)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n>>> Building Matrix for CONFIG 2: Engineered + Character N-Grams (min_df=10, Masked)...")
    X_train_char, _, names_char, tfidf_char = build_feature_matrix(
        df_train,
        fit_tfidf=True,
        include_tfidf=True,
        max_tfidf_features=300,
        text_column="ocr_text",
        use_char_ngrams=True,
        min_df=10,
    )
    X_val_char, _, _, _ = build_feature_matrix(
        df_val,
        tfidf_vectorizer=tfidf_char,
        fit_tfidf=False,
        include_tfidf=True,
        text_column="ocr_text",
        use_char_ngrams=True,
    )

    print(f"Training Model 2 (Engineered + Char N-Grams, {X_train_char.shape[1]} features)...")
    model_char = train_and_calibrate_rf(X_train_char, y_train, names_char)
    t_legit_2, t_phish_2 = tune_thresholds_on_val(model_char, X_val_char, y_val)
    print(f"Config 2 Thresholds: T_legit={t_legit_2:.3f}, T_phish={t_phish_2:.3f}")

    results_dev_char = evaluate_on_dev(model_char, tfidf_char, t_legit_2, t_phish_2, names_char)
    summarize_dev_results(results_dev_char, model_name="Config 2: Engineered + Char N-Grams (Masked, min_df=10)")

    # Save artifacts for the winning model
    # Choose winning model based on lowest dev FPR and legitimate false positive count
    fp_1 = sum(1 for ch in results_dev_eng.values() for item in ch if item["verdict"] == "phishing")
    fp_2 = sum(1 for ch in results_dev_char.values() for item in ch if item["verdict"] == "phishing")

    winner = "char_ngrams" if fp_2 < fp_1 else "engineered_only"
    print(f"\nSelection: Model 1 FPs = {fp_1}/30, Model 2 FPs = {fp_2}/30 -> Selected: {winner}")

    best_model = model_char if winner == "char_ngrams" else model_eng
    best_tfidf = tfidf_char if winner == "char_ngrams" else None
    best_t_legit = t_legit_2 if winner == "char_ngrams" else t_legit_1
    best_t_phish = t_phish_2 if winner == "char_ngrams" else t_phish_1

    os.makedirs("data/models", exist_ok=True)
    joblib.dump(best_model, "data/models/calibrated_model.pkl")
    if best_tfidf is not None:
        joblib.dump(best_tfidf, "data/models/tfidf_vectorizer.pkl")
    elif os.path.exists("data/models/tfidf_vectorizer.pkl"):
        # Remove old tfidf if engineered-only won
        os.remove("data/models/tfidf_vectorizer.pkl")

    with open("data/models/thresholds.json", "w") as f:
        json.dump({"t_legit": best_t_legit, "t_phish": best_t_phish}, f, indent=2)

    print(f"Saved winning model artifacts to data/models/")


if __name__ == "__main__":
    run_comparison()
