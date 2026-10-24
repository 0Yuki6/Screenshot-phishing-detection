"""
One-shot cross-validated training with real-data integration.

Steps:
1. Move 50% of data/real_holdout/final/<ch>/ into data/real_train/<ch>/legitimate/
2. Update leakage guard (remaining final + smishtank_test_holdout + wild_test)
3. Build combined training set (synthetic + real at 5x weight)
4. Stratified 5-fold CV → out-of-fold threshold tuning (FPR ≤ 5% on real legit)
5. Train final model on all data, save new artifacts
6. Evaluate old vs new on test holdouts
"""

import os, sys, hashlib, shutil, json, logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict, PredefinedSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import build_feature_matrix
from src.pipeline import PhishingPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SEED = 42


def file_hash(fp):
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def image_files(d):
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if os.path.splitext(f)[1].lower() in EXTS]


# ── STEP 1: Move 50% of real_holdout/final into real_train ──────────────
def split_holdout_final():
    rng = np.random.RandomState(SEED)
    moved = 0
    for ch in ["email", "sms", "whatsapp"]:
        src_dir = f"data/real_holdout/final/{ch}"
        dst_dir = f"data/real_train/{ch}/legitimate"
        os.makedirs(dst_dir, exist_ok=True)
        files = image_files(src_dir)
        if not files:
            continue
        # Hash-based deterministic shuffle
        hashes = [(f, file_hash(f)) for f in files]
        hashes.sort(key=lambda x: x[1])
        n_move = len(hashes) // 2
        idx = rng.permutation(len(hashes))
        to_move = [hashes[i] for i in idx[:n_move]]
        for fp, _ in to_move:
            dst = os.path.join(dst_dir, os.path.basename(fp))
            if not os.path.exists(dst):
                shutil.copy2(fp, dst)
                moved += 1
    logger.info("Moved %d images from real_holdout/final to real_train", moved)


# ── Collect blocked hashes (remaining final + smishtank_test_holdout + wild_test)
def get_blocked_hashes():
    blocked = set()
    for d in ["data/real_holdout/final", "data/smishtank_test_holdout", "data/wild_test"]:
        for fp in _walk_images(d):
            blocked.add(file_hash(fp))
    return blocked


def _walk_images(d):
    if not os.path.isdir(d):
        return
    for root, _, files in os.walk(d):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                yield os.path.join(root, f)


# ── STEP 2: Re-ingest real training data ────────────────────────────────
def reingest_real_data(blocked_hashes):
    from src.ocr import run_dual_variant_ocr
    from src.masking import strip_template_chrome
    import cv2

    records = []
    seen_hashes = set()
    channels = ["email", "sms", "whatsapp"]
    labels = ["legitimate", "phishing"]

    for ch in channels:
        for lbl in labels:
            folder = f"data/real_train/{ch}/{lbl}"
            if not os.path.isdir(folder):
                continue
            for fn in sorted(os.listdir(folder)):
                ext = os.path.splitext(fn)[1].lower()
                if ext not in EXTS:
                    continue
                fp = os.path.join(folder, fn)
                h = file_hash(fp)
                if h in blocked_hashes:
                    continue
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                img = cv2.imread(fp)
                if img is None:
                    continue
                ocr = run_dual_variant_ocr(img)
                clean = strip_template_chrome(ocr["text"], channel=ch)
                records.append({
                    "message_id": f"real_{ch}_{lbl}_{fn}",
                    "channel": ch,
                    "label": lbl,
                    "ocr_text": ocr["text"],
                    "clean_text": clean,
                    "confidence": float(ocr["confidence"]),
                    "winning_variant": ocr["winning_variant"],
                    "source": "real",
                    "file_hash": h,
                })

    df = pd.DataFrame(records)
    out = "data/processed/real_train_samples.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Re-ingested %d real training samples", len(df))
    return df


# ── STEP 3-5: Build combined set, CV, threshold, final model ───────────
def train_with_cv():
    # Load synthetic data
    syn_df = pd.read_csv("data/processed/dataset_with_groups.csv")
    syn_train = syn_df[syn_df["split"] == "train"].copy()
    syn_train["source"] = syn_train.get("source", "synthetic")
    if "source" not in syn_train.columns or syn_train["source"].isna().all():
        syn_train["source"] = "synthetic"
    syn_train["source"] = syn_train["source"].fillna("synthetic")

    # Load real data
    real_csv = "data/processed/real_train_samples.csv"
    if os.path.exists(real_csv):
        real_df = pd.read_csv(real_csv)
        real_df["source"] = "real"
        if "clean_text" not in real_df.columns:
            real_df["clean_text"] = real_df["ocr_text"]
    else:
        real_df = pd.DataFrame()

    # Combine with 5x real weight (duplicate real samples 5 times)
    if len(real_df) > 0:
        real_expanded = pd.concat([real_df] * 5, ignore_index=True)
        combined = pd.concat([syn_train, real_expanded], ignore_index=True)
    else:
        combined = syn_train.copy()

    # Remove exact duplicates by hash where available
    if "file_hash" in combined.columns:
        # Keep duplicates for weighting (real 5x), just dedupe within same source occurrence
        pass

    logger.info("Combined training set: %d samples (%d synthetic, %d real x5)",
                len(combined), len(syn_train), len(real_df))

    # Build stratification key
    combined["strat_key"] = (combined["channel"].fillna("unk") + "_" +
                             combined["label"].fillna("unk") + "_" +
                             combined["source"].fillna("unk"))

    # Check stratum sizes
    strat_counts = combined["strat_key"].value_counts()
    small_strata = strat_counts[strat_counts < 5]
    if len(small_strata) > 0:
        logger.warning("Strata with <5 items: %s", dict(small_strata))
        # Merge small strata into nearest
        for key in small_strata.index:
            combined.loc[combined["strat_key"] == key, "strat_key"] = key.rsplit("_", 1)[0] + "_merged"

    # Extract features (fit TF-IDF on full combined for CV)
    logger.info("Extracting features...")
    X_all, y_all, feat_names, tfidf_vec = build_feature_matrix(
        combined, fit_tfidf=True, max_tfidf_features=200
    )

    strat_labels = combined["strat_key"].values

    # Stratified 5-fold CV - out-of-fold predictions
    logger.info("Running stratified 5-fold cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    base_rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1
    )

    # Get out-of-fold probabilities
    oof_probs = np.zeros(len(y_all))
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_all, strat_labels)):
        X_tr, X_vl = X_all[train_idx], X_all[val_idx]
        y_tr, y_vl = y_all[train_idx], y_all[val_idx]

        rf_fold = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1
        )
        rf_fold.fit(X_tr, y_tr)
        oof_probs[val_idx] = rf_fold.predict_proba(X_vl)[:, 1]
        logger.info("Fold %d: val accuracy %.3f", fold_i + 1,
                     np.mean((oof_probs[val_idx] >= 0.5) == y_vl))

    # Threshold tuning on out-of-fold scores
    # Target: decided FPR ≤ 5% on REAL LEGITIMATE samples
    real_legit_mask = ((combined["source"] == "real") &
                       (combined["label"] == "legitimate")).values
    # Use first occurrence only (not 5x duplicates) for threshold tuning
    # Mark first-occurrence real samples
    if len(real_df) > 0:
        n_syn = len(syn_train)
        n_real_orig = len(real_df)
        # First copy of real data starts at n_syn, ends at n_syn + n_real_orig
        first_real_mask = np.zeros(len(combined), dtype=bool)
        first_real_mask[n_syn:n_syn + n_real_orig] = True
        real_legit_first = first_real_mask & (combined["label"].values == "legitimate")
    else:
        real_legit_first = real_legit_mask

    t_legit, t_phish = optimize_thresholds_cv(
        y_all, oof_probs, real_legit_first, target_fpr=0.05
    )
    logger.info("CV thresholds: T_legit=%.3f, T_phish=%.3f", t_legit, t_phish)

    # Train final model on ALL combined data with calibration
    logger.info("Training final calibrated model on all data...")
    # Use PredefinedSplit: use 80/20 for calibration
    rng = np.random.RandomState(SEED)
    cal_idx = rng.choice(len(y_all), size=len(y_all) // 5, replace=False)
    test_fold = np.full(len(y_all), -1)
    test_fold[cal_idx] = 0
    ps = PredefinedSplit(test_fold)

    final_base = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    final_model = CalibratedClassifierCV(estimator=final_base, method="sigmoid", cv=ps)
    final_model.fit(X_all, y_all)

    # Save artifacts under new names
    out_dir = "data/models"
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(final_model, os.path.join(out_dir, "calibrated_model_cv5.pkl"))
    joblib.dump(tfidf_vec, os.path.join(out_dir, "tfidf_vectorizer_cv5.pkl"))
    thresh = {"t_legit": t_legit, "t_phish": t_phish}
    with open(os.path.join(out_dir, "thresholds_cv5.json"), "w") as f:
        json.dump(thresh, f, indent=2)
    logger.info("Saved cv5 model artifacts")

    return final_model, tfidf_vec, t_legit, t_phish


def optimize_thresholds_cv(y, probs, real_legit_mask, target_fpr=0.05):
    best_tl, best_tp = 0.20, 0.75
    best_score = -1

    for tl in np.linspace(0.05, 0.45, 41):
        for tp in np.linspace(0.55, 0.95, 41):
            if tl >= tp:
                continue
            decided = (probs <= tl) | (probs >= tp)
            if np.sum(decided) == 0:
                continue

            # Overall metrics
            y_dec = y[decided]
            p_dec = (probs[decided] >= tp).astype(int)
            tn = np.sum((y_dec == 0) & (p_dec == 0))
            fp = np.sum((y_dec == 0) & (p_dec == 1))
            fn = np.sum((y_dec == 1) & (p_dec == 0))
            tp_c = np.sum((y_dec == 1) & (p_dec == 1))

            overall_fpr = fp / max(1, tn + fp)
            recall = tp_c / max(1, tp_c + fn)
            abstention = 1.0 - np.mean(decided)

            # Real legitimate FPR
            rl_decided = decided & real_legit_mask
            rl_total_decided = np.sum(rl_decided)
            if rl_total_decided > 0:
                rl_fp = np.sum((probs[rl_decided] >= tp))
                rl_fpr = rl_fp / rl_total_decided
            else:
                rl_fpr = 0.0

            if rl_fpr <= target_fpr and recall >= 0.90 and abstention < 0.15:
                score = recall * (1 - abstention) * (1 - rl_fpr)
                if score > best_score:
                    best_score = score
                    best_tl = round(float(tl), 3)
                    best_tp = round(float(tp), 3)

    return best_tl, best_tp


# ── STEP 6: Evaluate old vs new ─────────────────────────────────────────
def evaluate_comparison(new_model, new_tfidf, new_tl, new_tp):
    # Load old model
    old_model = joblib.load("data/models/calibrated_model.pkl")
    old_tfidf = joblib.load("data/models/tfidf_vectorizer.pkl")
    with open("data/models/thresholds.json") as f:
        old_thresh = json.load(f)
    old_tl, old_tp = old_thresh["t_legit"], old_thresh["t_phish"]

    results = {}

    # 1. Synthetic test split
    syn_df = pd.read_csv("data/processed/dataset_with_groups.csv")
    test_df = syn_df[syn_df["split"] == "test"].copy()

    for tag, model, tfidf, tl, tp in [
        ("old", old_model, old_tfidf, old_tl, old_tp),
        ("new", new_model, new_tfidf, new_tl, new_tp),
    ]:
        X, y, _, _ = build_feature_matrix(test_df, tfidf_vectorizer=tfidf, fit_tfidf=False)
        probs = model.predict_proba(X)[:, 1]
        decided = (probs <= tl) | (probs >= tp)
        y_dec = y[decided]
        p_dec = (probs[decided] >= tp).astype(int)
        fp = np.sum((y_dec == 0) & (p_dec == 1))
        tn = np.sum((y_dec == 0) & (p_dec == 0))
        tp_c = np.sum((y_dec == 1) & (p_dec == 1))
        fn = np.sum((y_dec == 1) & (p_dec == 0))
        fpr = fp / max(1, tn + fp)
        rec = tp_c / max(1, tp_c + fn)
        acc = (tn + tp_c) / max(1, len(y_dec))
        results[f"synth_test_{tag}"] = {
            "acc": round(acc, 3), "fpr": round(fpr, 3), "recall": round(rec, 3),
            "abstain": round(1 - np.mean(decided), 3), "n": len(y)
        }

    # 2. Remaining real holdout/final (the 50% not moved to train)
    pipe_old = PhishingPipeline()
    pipe_new = PhishingPipeline(
        model_path="data/models/calibrated_model_cv5.pkl",
        tfidf_path="data/models/tfidf_vectorizer_cv5.pkl",
        thresholds_path="data/models/thresholds_cv5.json",
    )

    for tag, pipe, tl_val, tp_val in [("old", pipe_old, old_tl, old_tp), ("new", pipe_new, new_tl, new_tp)]:
        fp_count = 0
        legit_count = 0
        unc_count = 0
        total = 0
        for ch in ["email", "sms", "whatsapp"]:
            ch_dir = f"data/real_holdout/final/{ch}"
            for img_fp in image_files(ch_dir):
                total += 1
                try:
                    res = pipe.predict_image(img_fp)
                    v = res["verdict"]
                    if v == "phishing":
                        fp_count += 1
                    elif v == "legitimate":
                        legit_count += 1
                    else:
                        unc_count += 1
                except Exception:
                    unc_count += 1
        decided = fp_count + legit_count
        fpr = fp_count / max(1, decided)
        results[f"real_final_{tag}"] = {
            "total": total, "fp": fp_count, "legit": legit_count,
            "unc": unc_count, "fpr": round(fpr, 3)
        }

    # 3. Smishtank test holdout
    for tag, pipe in [("old", pipe_old), ("new", pipe_new)]:
        tp_c = 0
        fn_c = 0
        unc = 0
        total = 0
        smish_dir = "data/smishtank_test_holdout"
        for img_fp in image_files(smish_dir):
            total += 1
            try:
                res = pipe.predict_image(img_fp)
                v = res["verdict"]
                if v == "phishing":
                    tp_c += 1
                elif v == "legitimate":
                    fn_c += 1
                else:
                    unc += 1
            except Exception:
                unc += 1
        decided = tp_c + fn_c
        rec = tp_c / max(1, decided)
        results[f"smishtank_{tag}"] = {
            "total": total, "tp": tp_c, "fn": fn_c, "unc": unc, "recall": round(rec, 3)
        }

    return results


def main():
    # Step 1
    split_holdout_final()

    # Step 2: get blocked hashes and re-ingest
    blocked = get_blocked_hashes()
    logger.info("Blocked %d holdout hashes", len(blocked))
    reingest_real_data(blocked)

    # Steps 3-5
    model, tfidf, tl, tp = train_with_cv()

    # Step 6
    results = evaluate_comparison(model, tfidf, tl, tp)

    # Print concise summary (max 6 lines)
    print(f"\nThresholds: T_legit={tl:.3f}, T_phish={tp:.3f}")
    so = results.get("synth_test_old", {})
    sn = results.get("synth_test_new", {})
    print(f"Synth test: OLD acc={so.get('acc')}/fpr={so.get('fpr')}/rec={so.get('recall')} | NEW acc={sn.get('acc')}/fpr={sn.get('fpr')}/rec={sn.get('recall')}")
    ro = results.get("real_final_old", {})
    rn = results.get("real_final_new", {})
    print(f"Real final: OLD fp={ro.get('fp')}/{ro.get('total')} fpr={ro.get('fpr')} | NEW fp={rn.get('fp')}/{rn.get('total')} fpr={rn.get('fpr')}")
    smo = results.get("smishtank_old", {})
    smn = results.get("smishtank_new", {})
    print(f"Smishtank:  OLD rec={smo.get('recall')} ({smo.get('tp')}/{smo.get('total')}) | NEW rec={smn.get('recall')} ({smn.get('tp')}/{smn.get('total')})")


if __name__ == "__main__":
    main()
