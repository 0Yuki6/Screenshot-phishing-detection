"""
Quick diagnostic: check OOF score distribution for real legit samples,
then re-tune thresholds with wider gap constraint and re-evaluate.
"""
import os, sys, json, logging
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import build_feature_matrix
from src.pipeline import PhishingPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
SEED = 42
EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def image_files(d):
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if os.path.splitext(f)[1].lower() in EXTS]


def main():
    # Rebuild combined training set (same as train_cv.py, no OCR needed)
    syn_df = pd.read_csv("data/processed/dataset_with_groups.csv")
    syn_train = syn_df[syn_df["split"] == "train"].copy()
    syn_train["source"] = syn_train["source"].fillna("synthetic") if "source" in syn_train.columns else "synthetic"

    real_df = pd.read_csv("data/processed/real_train_samples.csv")
    real_df["source"] = "real"
    if "clean_text" not in real_df.columns:
        real_df["clean_text"] = real_df["ocr_text"]

    real_expanded = pd.concat([real_df] * 5, ignore_index=True)
    combined = pd.concat([syn_train, real_expanded], ignore_index=True)

    combined["strat_key"] = (combined["channel"].fillna("unk") + "_" +
                             combined["label"].fillna("unk") + "_" +
                             combined["source"].fillna("unk"))
    strat_counts = combined["strat_key"].value_counts()
    small_strata = strat_counts[strat_counts < 5]
    for key in small_strata.index:
        combined.loc[combined["strat_key"] == key, "strat_key"] = key.rsplit("_", 1)[0] + "_merged"

    X_all, y_all, feat_names, tfidf_vec = build_feature_matrix(
        combined, fit_tfidf=True, max_tfidf_features=200
    )
    strat_labels = combined["strat_key"].values

    # Re-run OOF
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(y_all))
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_all, strat_labels)):
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     random_state=SEED, n_jobs=-1)
        rf.fit(X_all[train_idx], y_all[train_idx])
        oof_probs[val_idx] = rf.predict_proba(X_all[val_idx])[:, 1]

    # Diagnostic: score distribution for real legit (first copy only)
    n_syn = len(syn_train)
    n_real = len(real_df)
    first_real = np.zeros(len(combined), dtype=bool)
    first_real[n_syn:n_syn + n_real] = True
    real_legit_first = first_real & (combined["label"].values == "legitimate")
    real_phish_first = first_real & (combined["label"].values == "phishing")
    syn_legit = (~first_real) & (combined["label"].values == "legitimate") & (np.arange(len(combined)) < n_syn)
    syn_phish = (~first_real) & (combined["label"].values == "phishing") & (np.arange(len(combined)) < n_syn)

    print("\n=== OOF Score Distribution ===")
    for name, mask in [("Syn legit", syn_legit), ("Syn phish", syn_phish),
                       ("Real legit", real_legit_first), ("Real phish", real_phish_first)]:
        scores = oof_probs[mask]
        if len(scores) > 0:
            print(f"  {name:12s}: n={len(scores):4d}  min={scores.min():.3f}  "
                  f"p25={np.percentile(scores,25):.3f}  med={np.median(scores):.3f}  "
                  f"p75={np.percentile(scores,75):.3f}  max={scores.max():.3f}")

    # Better threshold optimization:
    # 1. Require minimum gap of 0.25 between t_legit and t_phish
    # 2. Target FPR <= 5% on ALL legitimate OOF scores (not just real)
    # 3. Prefer wider gap (more uncertainty = safer)
    print("\n=== Threshold Re-optimization (min gap 0.25) ===")
    best_tl, best_tp = 0.20, 0.75
    best_score = -1

    all_legit_mask = (y_all == 0)

    for tl in np.linspace(0.05, 0.40, 36):
        for tp in np.linspace(0.60, 0.95, 36):
            gap = tp - tl
            if gap < 0.25:
                continue

            decided = (oof_probs <= tl) | (oof_probs >= tp)
            if np.sum(decided) == 0:
                continue

            y_dec = y_all[decided]
            p_dec = (oof_probs[decided] >= tp).astype(int)
            tn = np.sum((y_dec == 0) & (p_dec == 0))
            fp = np.sum((y_dec == 0) & (p_dec == 1))
            fn = np.sum((y_dec == 1) & (p_dec == 0))
            tp_c = np.sum((y_dec == 1) & (p_dec == 1))

            fpr = fp / max(1, tn + fp)
            recall = tp_c / max(1, tp_c + fn)
            abstention = 1.0 - np.mean(decided)

            # Real legit FPR
            rl_dec = decided & real_legit_first
            rl_n = np.sum(rl_dec)
            if rl_n > 0:
                rl_fp = np.sum(oof_probs[rl_dec] >= tp)
                rl_fpr = rl_fp / rl_n
            else:
                rl_fpr = 0.0

            # Constraints: overall FPR <= 2%, real legit FPR <= 5%, recall >= 90%
            if fpr <= 0.02 and rl_fpr <= 0.05 and recall >= 0.90 and abstention < 0.15:
                # Score: prefer wider gap (more conservative), higher recall
                score = recall * (1 - abstention) * gap
                if score > best_score:
                    best_score = score
                    best_tl = round(float(tl), 3)
                    best_tp = round(float(tp), 3)

    print(f"  Best: T_legit={best_tl}, T_phish={best_tp} (gap={best_tp-best_tl:.3f})")

    # Check OOF metrics with these thresholds
    decided = (oof_probs <= best_tl) | (oof_probs >= best_tp)
    y_dec = y_all[decided]
    p_dec = (oof_probs[decided] >= best_tp).astype(int)
    tn = np.sum((y_dec == 0) & (p_dec == 0))
    fp = np.sum((y_dec == 0) & (p_dec == 1))
    fn = np.sum((y_dec == 1) & (p_dec == 0))
    tp_c = np.sum((y_dec == 1) & (p_dec == 1))
    print(f"  OOF: acc={((tn+tp_c)/max(1,len(y_dec))):.3f} fpr={fp/max(1,tn+fp):.3f} "
          f"rec={tp_c/max(1,tp_c+fn):.3f} abst={1-np.mean(decided):.3f}")

    rl_dec = decided & real_legit_first
    rl_fp = np.sum(oof_probs[rl_dec] >= best_tp) if np.sum(rl_dec) > 0 else 0
    print(f"  Real legit OOF: {rl_fp}/{np.sum(rl_dec)} FP (FPR={rl_fp/max(1,np.sum(rl_dec)):.3f})")

    # Save updated thresholds
    thresh = {"t_legit": best_tl, "t_phish": best_tp}
    with open("data/models/thresholds_cv5.json", "w") as f:
        json.dump(thresh, f, indent=2)

    # Quick eval on holdout using new pipeline with updated thresholds
    pipe_new = PhishingPipeline(
        model_path="data/models/calibrated_model_cv5.pkl",
        tfidf_path="data/models/tfidf_vectorizer_cv5.pkl",
        thresholds_path="data/models/thresholds_cv5.json",
    )
    pipe_old = PhishingPipeline()

    print("\n=== Holdout Evaluation ===")
    for tag, pipe in [("OLD", pipe_old), ("NEW", pipe_new)]:
        fp_c = leg_c = unc_c = 0
        for ch in ["email", "sms", "whatsapp"]:
            for img_fp in image_files(f"data/real_holdout/final/{ch}"):
                try:
                    r = pipe.predict_image(img_fp)
                    v = r["verdict"]
                    if v == "phishing": fp_c += 1
                    elif v == "legitimate": leg_c += 1
                    else: unc_c += 1
                except: unc_c += 1
        total = fp_c + leg_c + unc_c
        dec = fp_c + leg_c
        print(f"  {tag} real_final: fp={fp_c} leg={leg_c} unc={unc_c} total={total} "
              f"fpr={fp_c/max(1,dec):.3f}")

    for tag, pipe in [("OLD", pipe_old), ("NEW", pipe_new)]:
        tp_c = fn_c = unc_c = 0
        for img_fp in image_files("data/smishtank_test_holdout"):
            try:
                r = pipe.predict_image(img_fp)
                v = r["verdict"]
                if v == "phishing": tp_c += 1
                elif v == "legitimate": fn_c += 1
                else: unc_c += 1
            except: unc_c += 1
        dec = tp_c + fn_c
        print(f"  {tag} smishtank:  tp={tp_c} fn={fn_c} unc={unc_c} "
              f"rec={tp_c/max(1,dec):.3f}")


if __name__ == "__main__":
    main()
