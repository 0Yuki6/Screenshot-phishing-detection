"""
Stratified 5-Fold Cross-Validation, Threshold Fitting, Final Model Training, and Evaluation.
"""
import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, PredefinedSplit
from sklearn.metrics import f1_score, accuracy_score

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import build_feature_matrix, extract_features, FEATURE_NAMES
from src.masking import mask_text, strip_template_chrome
from src.rules import evaluate_rules
from src.channel_detector import detect_channel
from src.ocr import run_dual_variant_ocr, check_unreadable_gate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def file_hash(fp: Path) -> str:
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline_inference(
    raw_text: str,
    ocr_conf: float,
    channel: str,
    clean_text: str,
    model,
    tfidf_vec,
    t_legit: float,
    t_phish: float,
) -> str:
    """Run lightweight pipeline inference without SHAP to determine verdict."""
    is_unreadable, _ = check_unreadable_gate(
        text=clean_text,
        confidence=ocr_conf,
        min_text_len=15,
        min_confidence=35.0,
    )
    if is_unreadable:
        return "uncertain"

    scalar_dict = extract_features(clean_text, channel, allow_network=False, skip_whois=True)
    scalar_vector = np.array([[scalar_dict[name] for name in FEATURE_NAMES]], dtype=np.float64)
    masked_text = mask_text(clean_text)

    if tfidf_vec is not None:
        tfidf_feat = tfidf_vec.transform([masked_text])
        X = hstack([csr_matrix(scalar_vector), tfidf_feat]).toarray()
    else:
        X = scalar_vector

    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    phish_idx = classes.index(1) if 1 in classes else -1
    raw_ml_score = float(proba[phish_idx]) if phish_idx >= 0 else 0.5

    rule_eval = evaluate_rules(
        raw_ocr_text=raw_text,
        clean_text=clean_text,
        ocr_confidence=ocr_conf,
        ml_score=raw_ml_score,
        channel=channel,
        t_legit=t_legit,
        t_phish=t_phish,
    )
    return rule_eval["verdict"]


def main():
    # 1. Load Data
    syn_df = pd.read_csv("data/processed/dataset_with_groups.csv")
    syn_train = syn_df[syn_df["split"] == "train"].copy()
    syn_train["source"] = "synthetic"
    syn_train["clean_text"] = syn_train["clean_text"].fillna("")

    real_df = pd.read_csv("data/processed/real_train_samples.csv")
    real_df["source"] = "real"
    real_df["clean_text"] = real_df["clean_text"].fillna("")

    # Verify holdout leakage
    holdout_hashes = set()
    for d in ["data/real_holdout/final", "data/smishtank_test_holdout", "data/wild_test"]:
        if os.path.isdir(d):
            for p in Path(d).rglob("*"):
                if p.suffix.lower() in EXTS:
                    holdout_hashes.add(file_hash(p))

    # Deduplicate unique training samples
    combined_unique = pd.concat([syn_train, real_df], ignore_index=True)
    if "file_hash" in combined_unique.columns:
        # Drop duplicates where file_hash is non-null
        has_hash = combined_unique["file_hash"].notna() & (combined_unique["file_hash"] != "")
        hashes_seen = set()
        keep_mask = []
        for idx, row in combined_unique.iterrows():
            h = row.get("file_hash")
            if pd.notna(h) and h:
                if h in holdout_hashes:
                    logger.warning("Dropping leaked hash in training: %s", h)
                    keep_mask.append(False)
                elif h in hashes_seen:
                    keep_mask.append(False)
                else:
                    hashes_seen.add(h)
                    keep_mask.append(True)
            else:
                keep_mask.append(True)
        combined_unique = combined_unique[keep_mask].reset_index(drop=True)

    # Ingested counts
    real_counts = real_df.groupby(["channel", "label"]).size().to_dict()
    syn_counts = syn_train.groupby(["channel", "label"]).size().to_dict()

    # Stratification key: channel + label + source
    combined_unique["strat_key"] = (
        combined_unique["channel"].fillna("unk") + "_" +
        combined_unique["label"].fillna("unk") + "_" +
        combined_unique["source"].fillna("unk")
    )
    strata_counts = combined_unique["strat_key"].value_counts()
    small_strata = strata_counts[strata_counts < 5]
    if len(small_strata) > 0:
        print(f"CRITICAL: Stratum has fewer than 5 items: {dict(small_strata)}")
        sys.exit(1)

    # 2. Build feature matrix for combined_unique
    logger.info("Extracting features for %d unique samples...", len(combined_unique))
    X_u, y_u, feat_names, tfidf_vec = build_feature_matrix(
        combined_unique, fit_tfidf=True, max_tfidf_features=200
    )
    strat_labels = combined_unique["strat_key"].values
    is_real = (combined_unique["source"] == "real").values
    is_real_legit = ((combined_unique["source"] == "real") & (combined_unique["label"] == "legitimate")).values

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(y_u))
    fold_f1_scores = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_u, strat_labels)):
        # 5x weight for real samples in training fold
        train_is_real = is_real[train_idx]
        real_indices = train_idx[train_is_real]
        expanded_train_idx = np.concatenate([train_idx] + [real_indices] * 4)

        X_tr = X_u[expanded_train_idx]
        y_tr = y_u[expanded_train_idx]
        X_vl = X_u[val_idx]
        y_vl = y_u[val_idx]

        rf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        val_p = rf.predict_proba(X_vl)[:, 1]
        oof_probs[val_idx] = val_p

        val_pred = (val_p >= 0.5).astype(int)
        fold_f1 = f1_score(y_vl, val_pred)
        fold_f1_scores.append(fold_f1)

    f1_mean = float(np.mean(fold_f1_scores))
    f1_std = float(np.std(fold_f1_scores))
    logger.info("5-fold CV F1: %.4f +/- %.4f", f1_mean, f1_std)

    # 3. Fit thresholds using OOF scores
    # Target: decided FPR <= 5% on real legitimate samples
    best_tl, best_tp = 0.20, 0.75
    best_score = -1

    for tl in np.linspace(0.05, 0.45, 41):
        for tp in np.linspace(0.55, 0.95, 41):
            if tp - tl < 0.20:
                continue
            decided = (oof_probs <= tl) | (oof_probs >= tp)
            if np.sum(decided) == 0:
                continue

            y_dec = y_u[decided]
            p_dec = (oof_probs[decided] >= tp).astype(int)
            tn = np.sum((y_dec == 0) & (p_dec == 0))
            fp = np.sum((y_dec == 0) & (p_dec == 1))
            fn = np.sum((y_dec == 1) & (p_dec == 0))
            tp_c = np.sum((y_dec == 1) & (p_dec == 1))

            overall_fpr = fp / max(1, tn + fp)
            rec = tp_c / max(1, tp_c + fn)
            abst = 1.0 - np.mean(decided)

            # Real legit FPR
            rl_dec = decided & is_real_legit
            rl_total = np.sum(rl_dec)
            if rl_total > 0:
                rl_fp = np.sum(oof_probs[rl_dec] >= tp)
                rl_fpr = rl_fp / rl_total
            else:
                rl_fpr = 0.0

            if rl_fpr <= 0.05 and overall_fpr <= 0.02 and rec >= 0.90 and abst < 0.20:
                sc = rec * (1 - abst) * (1 - rl_fpr) * (tp - tl)
                if sc > best_score:
                    best_score = sc
                    best_tl = round(float(tl), 3)
                    best_tp = round(float(tp), 3)

    logger.info("Optimal thresholds: T_legit=%.3f, T_phish=%.3f (score=%.4f)", best_tl, best_tp, best_score)

    # 4. Fit final model on all training data with real at 5x weight
    logger.info("Fitting final model on all training data...")
    real_indices_all = np.where(is_real)[0]
    all_train_indices = np.concatenate([np.arange(len(y_u))] + [real_indices_all] * 4)

    X_train_final = X_u[all_train_indices]
    y_train_final = y_u[all_train_indices]

    # Sigmoid calibration with PredefinedSplit on a 20% holdout split of training data
    rng = np.random.RandomState(SEED)
    cal_idx = rng.choice(len(y_train_final), size=len(y_train_final) // 5, replace=False)
    test_fold = np.full(len(y_train_final), -1)
    test_fold[cal_idx] = 0
    ps = PredefinedSplit(test_fold)

    base_rf_final = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    final_model = CalibratedClassifierCV(estimator=base_rf_final, method="sigmoid", cv=ps)
    final_model.fit(X_train_final, y_train_final)

    # Save under new filenames (do NOT overwrite served model)
    out_dir = "data/models"
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(final_model, os.path.join(out_dir, "calibrated_model_cv5.pkl"))
    joblib.dump(tfidf_vec, os.path.join(out_dir, "tfidf_vectorizer_cv5.pkl"))
    with open(os.path.join(out_dir, "thresholds_cv5.json"), "w") as f:
        json.dump({"t_legit": best_tl, "t_phish": best_tp}, f, indent=2)
    with open(os.path.join(out_dir, "feature_names_cv5.json"), "w") as f:
        json.dump(feat_names, f, indent=2)

    # 5. Evaluate Old vs New Model
    old_model = joblib.load(os.path.join(out_dir, "calibrated_model.pkl"))
    old_tfidf = joblib.load(os.path.join(out_dir, "tfidf_vectorizer.pkl"))
    with open(os.path.join(out_dir, "thresholds.json"), "r") as f:
        old_thresh = json.load(f)
    old_tl = float(old_thresh["t_legit"])
    old_tp = float(old_thresh["t_phish"])

    # 5a. Synthetic Test Split
    syn_test = syn_df[syn_df["split"] == "test"].copy()
    syn_test["clean_text"] = syn_test["clean_text"].fillna("")

    # Old on synth test
    X_test_old, y_test_old, _, _ = build_feature_matrix(
        syn_test, tfidf_vectorizer=old_tfidf, fit_tfidf=False
    )
    p_test_old = old_model.predict_proba(X_test_old)[:, 1]
    dec_test_old = (p_test_old <= old_tl) | (p_test_old >= old_tp)
    cov_test_old = float(np.mean(dec_test_old))
    acc_test_old = float(accuracy_score(y_test_old[dec_test_old], (p_test_old[dec_test_old] >= old_tp).astype(int))) if np.any(dec_test_old) else 0.0

    # New on synth test
    X_test_new, y_test_new, _, _ = build_feature_matrix(
        syn_test, tfidf_vectorizer=tfidf_vec, fit_tfidf=False
    )
    p_test_new = final_model.predict_proba(X_test_new)[:, 1]
    dec_test_new = (p_test_new <= best_tl) | (p_test_new >= best_tp)
    cov_test_new = float(np.mean(dec_test_new))
    acc_test_new = float(accuracy_score(y_test_new[dec_test_new], (p_test_new[dec_test_new] >= best_tp).astype(int))) if np.any(dec_test_new) else 0.0

    # 5b. Remaining Real Final Holdout
    import cv2
    final_imgs = []
    for ch in ["email", "sms", "whatsapp"]:
        ch_dir = Path(f"data/real_holdout/final/{ch}")
        if ch_dir.is_dir():
            for p in sorted(ch_dir.glob("*")):
                if p.suffix.lower() in EXTS:
                    final_imgs.append((p, ch, "legitimate"))

    # Extract OCR once per image
    final_records = []
    for p, ch, lbl in final_imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        ocr_res = run_dual_variant_ocr(img)
        raw_t = ocr_res["text"]
        conf = float(ocr_res["confidence"])
        clean_t = strip_template_chrome(raw_t, channel=ch)
        final_records.append({
            "path": p, "channel": ch, "label": lbl,
            "raw_text": raw_t, "clean_text": clean_t, "confidence": conf
        })

    # Predict Old & New on Real Final
    old_final_correct, old_final_decided = 0, 0
    new_final_correct, new_final_decided = 0, 0
    total_final = len(final_records)

    for rec in final_records:
        v_old = run_pipeline_inference(
            rec["raw_text"], rec["confidence"], rec["channel"], rec["clean_text"],
            old_model, old_tfidf, old_tl, old_tp
        )
        if v_old != "uncertain":
            old_final_decided += 1
            if v_old == rec["label"]:
                old_final_correct += 1

        v_new = run_pipeline_inference(
            rec["raw_text"], rec["confidence"], rec["channel"], rec["clean_text"],
            final_model, tfidf_vec, best_tl, best_tp
        )
        if v_new != "uncertain":
            new_final_decided += 1
            if v_new == rec["label"]:
                new_final_correct += 1

    acc_final_old = old_final_correct / max(1, old_final_decided)
    cov_final_old = old_final_decided / max(1, total_final)
    acc_final_new = new_final_correct / max(1, new_final_decided)
    cov_final_new = new_final_decided / max(1, total_final)

    # 5c. Smishtank Test Holdout
    smish_imgs = []
    smish_dir = Path("data/smishtank_test_holdout")
    if smish_dir.is_dir():
        for p in sorted(smish_dir.glob("*")):
            if p.suffix.lower() in EXTS:
                smish_imgs.append((p, "sms", "phishing"))

    smish_records = []
    for p, ch, lbl in smish_imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        ocr_res = run_dual_variant_ocr(img)
        raw_t = ocr_res["text"]
        conf = float(ocr_res["confidence"])
        clean_t = strip_template_chrome(raw_t, channel=ch)
        smish_records.append({
            "path": p, "channel": ch, "label": lbl,
            "raw_text": raw_t, "clean_text": clean_t, "confidence": conf
        })

    old_smish_correct, old_smish_decided = 0, 0
    new_smish_correct, new_smish_decided = 0, 0
    total_smish = len(smish_records)

    for rec in smish_records:
        v_old = run_pipeline_inference(
            rec["raw_text"], rec["confidence"], rec["channel"], rec["clean_text"],
            old_model, old_tfidf, old_tl, old_tp
        )
        if v_old != "uncertain":
            old_smish_decided += 1
            if v_old == rec["label"]:
                old_smish_correct += 1

        v_new = run_pipeline_inference(
            rec["raw_text"], rec["confidence"], rec["channel"], rec["clean_text"],
            final_model, tfidf_vec, best_tl, best_tp
        )
        if v_new != "uncertain":
            new_smish_decided += 1
            if v_new == rec["label"]:
                new_smish_correct += 1

    acc_smish_old = old_smish_correct / max(1, old_smish_decided)
    cov_smish_old = old_smish_decided / max(1, total_smish)
    acc_smish_new = new_smish_correct / max(1, new_smish_decided)
    cov_smish_new = new_smish_decided / max(1, total_smish)

    # 6. EXACT 6-LINE OUTPUT (as specified in instructions)
    # Line 1: Ingested counts per channel/label/source
    line1 = (
        f"Ingested: email/legit/real={real_counts.get(('email', 'legitimate'), 0)}, "
        f"sms/legit/real={real_counts.get(('sms', 'legitimate'), 0)}, "
        f"sms/phish/real={real_counts.get(('sms', 'phishing'), 0)}, "
        f"whatsapp/legit/real={real_counts.get(('whatsapp', 'legitimate'), 0)}; "
        f"syn_train={len(syn_train)} (email={syn_counts.get(('email', 'legitimate'), 0) + syn_counts.get(('email', 'phishing'), 0)}, "
        f"sms={syn_counts.get(('sms', 'legitimate'), 0) + syn_counts.get(('sms', 'phishing'), 0)}, "
        f"whatsapp={syn_counts.get(('whatsapp', 'legitimate'), 0) + syn_counts.get(('whatsapp', 'phishing'), 0)})"
    )
    # Line 2: CV F1 mean +/- std
    line2 = f"CV 5-fold F1: {f1_mean:.4f} +/- {f1_std:.4f}"
    # Line 3: New thresholds
    line3 = f"Thresholds: T_legit = {best_tl:.3f}, T_phish = {best_tp:.3f}"
    # Line 4: Synthetic test split
    line4 = f"Synthetic test: OLD acc={acc_test_old * 100:.1f}%, cov={cov_test_old * 100:.1f}% | NEW acc={acc_test_new * 100:.1f}%, cov={cov_test_new * 100:.1f}%"
    # Line 5: Remaining real final
    line5 = f"Remaining real final: OLD acc={acc_final_old * 100:.1f}%, cov={cov_final_old * 100:.1f}% | NEW acc={acc_final_new * 100:.1f}%, cov={cov_final_new * 100:.1f}%"
    # Line 6: Smishtank test holdout
    line6 = f"Smishtank holdout: OLD acc={acc_smish_old * 100:.1f}%, cov={cov_smish_old * 100:.1f}% | NEW acc={acc_smish_new * 100:.1f}%, cov={cov_smish_new * 100:.1f}%"

    print("\n" + "=" * 50)
    print(line1)
    print(line2)
    print(line3)
    print(line4)
    print(line5)
    print(line6)
    print("=" * 50)


if __name__ == "__main__":
    main()
