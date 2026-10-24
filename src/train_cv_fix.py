"""
Fix: Actually MOVE files from holdout/final to real_train (delete from source),
then re-run ingestion + CV training + evaluation.
"""
import os, sys, hashlib, shutil, json, logging
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def fix_split():
    """Find files that exist in both holdout/final and real_train, delete from holdout."""
    removed = 0
    for ch in ["email", "sms", "whatsapp"]:
        src_dir = f"data/real_holdout/final/{ch}"
        dst_dir = f"data/real_train/{ch}/legitimate"
        if not os.path.isdir(dst_dir):
            continue
        dst_files = set(os.listdir(dst_dir))
        for fn in sorted(os.listdir(src_dir)):
            if os.path.splitext(fn)[1].lower() not in EXTS:
                continue
            if fn in dst_files:
                # This file was copied but not removed — remove from source
                os.remove(os.path.join(src_dir, fn))
                removed += 1
    logger.info("Removed %d already-copied files from real_holdout/final", removed)

    # Verify counts
    for ch in ["email", "sms", "whatsapp"]:
        src = len(image_files(f"data/real_holdout/final/{ch}"))
        dst = len(image_files(f"data/real_train/{ch}/legitimate"))
        logger.info("  %s: holdout=%d, train_legit=%d", ch, src, dst)


def get_blocked_hashes():
    blocked = set()
    for d in ["data/real_holdout/final", "data/smishtank_test_holdout", "data/wild_test"]:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if os.path.splitext(f)[1].lower() in EXTS:
                    blocked.add(file_hash(os.path.join(root, f)))
    return blocked


def reingest(blocked):
    from src.ocr import run_dual_variant_ocr
    from src.masking import strip_template_chrome
    import cv2

    records = []
    seen = set()
    for ch in ["email", "sms", "whatsapp"]:
        for lbl in ["legitimate", "phishing"]:
            folder = f"data/real_train/{ch}/{lbl}"
            if not os.path.isdir(folder):
                continue
            for fn in sorted(os.listdir(folder)):
                if os.path.splitext(fn)[1].lower() not in EXTS:
                    continue
                fp = os.path.join(folder, fn)
                h = file_hash(fp)
                if h in blocked or h in seen:
                    continue
                seen.add(h)
                img = cv2.imread(fp)
                if img is None:
                    continue
                ocr = run_dual_variant_ocr(img)
                clean = strip_template_chrome(ocr["text"], channel=ch)
                records.append({
                    "message_id": f"real_{ch}_{lbl}_{fn}",
                    "channel": ch, "label": lbl,
                    "ocr_text": ocr["text"], "clean_text": clean,
                    "confidence": float(ocr["confidence"]),
                    "winning_variant": ocr["winning_variant"],
                    "source": "real", "file_hash": h,
                })
    df = pd.DataFrame(records)
    df.to_csv("data/processed/real_train_samples.csv", index=False)
    logger.info("Ingested %d real samples (%s)",
                len(df), dict(df.groupby(["channel","label"]).size()))
    return df


def train_and_eval():
    from src.features import build_feature_matrix
    from src.pipeline import PhishingPipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, PredefinedSplit

    syn_df = pd.read_csv("data/processed/dataset_with_groups.csv")
    syn_train = syn_df[syn_df["split"] == "train"].copy()
    syn_train["source"] = syn_train["source"].fillna("synthetic") if "source" in syn_train.columns else "synthetic"

    real_df = pd.read_csv("data/processed/real_train_samples.csv")
    real_df["source"] = "real"
    if "clean_text" not in real_df.columns:
        real_df["clean_text"] = real_df["ocr_text"]

    real_exp = pd.concat([real_df] * 5, ignore_index=True)
    combined = pd.concat([syn_train, real_exp], ignore_index=True)
    n_syn = len(syn_train)
    n_real = len(real_df)

    combined["strat_key"] = (combined["channel"].fillna("unk") + "_" +
                             combined["label"].fillna("unk") + "_" +
                             combined["source"].fillna("unk"))
    strat_counts = combined["strat_key"].value_counts()
    for key in strat_counts[strat_counts < 5].index:
        combined.loc[combined["strat_key"] == key, "strat_key"] = key.rsplit("_", 1)[0] + "_merged"

    logger.info("Combined: %d (%d syn + %d real x5)", len(combined), n_syn, n_real)

    X, y, fnames, tfidf = build_feature_matrix(combined, fit_tfidf=True, max_tfidf_features=200)
    strat = combined["strat_key"].values

    # 5-fold CV OOF
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for fi, (ti, vi) in enumerate(skf.split(X, strat)):
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     random_state=SEED, n_jobs=-1)
        rf.fit(X[ti], y[ti])
        oof[vi] = rf.predict_proba(X[vi])[:, 1]
        logger.info("Fold %d: acc=%.3f", fi+1, np.mean((oof[vi]>=0.5)==y[vi]))

    # Score distribution
    first_real = np.zeros(len(combined), dtype=bool)
    first_real[n_syn:n_syn+n_real] = True
    rl_mask = first_real & (combined["label"].values == "legitimate")
    rp_mask = first_real & (combined["label"].values == "phishing")
    sl_mask = (np.arange(len(combined)) < n_syn) & (y == 0)
    sp_mask = (np.arange(len(combined)) < n_syn) & (y == 1)

    print("\n=== OOF Scores ===")
    for name, m in [("Syn legit", sl_mask), ("Syn phish", sp_mask),
                    ("Real legit", rl_mask), ("Real phish", rp_mask)]:
        s = oof[m]
        if len(s):
            print(f"  {name:12s}: n={len(s):4d} min={s.min():.3f} med={np.median(s):.3f} "
                  f"p75={np.percentile(s,75):.3f} max={s.max():.3f}")

    # Threshold optimization (min gap 0.25, FPR<=2%, real legit FPR<=5%)
    best_tl, best_tp, best_sc = 0.20, 0.75, -1
    for tl in np.linspace(0.05, 0.40, 36):
        for tp in np.linspace(0.60, 0.95, 36):
            if tp - tl < 0.25:
                continue
            dec = (oof <= tl) | (oof >= tp)
            if not np.any(dec):
                continue
            yd = y[dec]; pd_ = (oof[dec] >= tp).astype(int)
            tn = np.sum((yd==0)&(pd_==0)); fp = np.sum((yd==0)&(pd_==1))
            fn = np.sum((yd==1)&(pd_==0)); tpc = np.sum((yd==1)&(pd_==1))
            fpr = fp/max(1,tn+fp); rec = tpc/max(1,tpc+fn)
            abst = 1-np.mean(dec)
            rld = dec & rl_mask; rln = np.sum(rld)
            rl_fpr = np.sum(oof[rld]>=tp)/max(1,rln) if rln else 0
            if fpr<=0.02 and rl_fpr<=0.05 and rec>=0.90 and abst<0.15:
                sc = rec*(1-abst)*(tp-tl)
                if sc > best_sc:
                    best_sc=sc; best_tl=round(float(tl),3); best_tp=round(float(tp),3)

    print(f"\nThresholds: T_legit={best_tl}, T_phish={best_tp}")

    # Final model
    rng = np.random.RandomState(SEED)
    ci = rng.choice(len(y), size=len(y)//5, replace=False)
    tf = np.full(len(y), -1); tf[ci] = 0
    ps = PredefinedSplit(tf)
    fm = CalibratedClassifierCV(
        estimator=RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                          random_state=SEED, n_jobs=-1),
        method="sigmoid", cv=ps)
    fm.fit(X, y)

    out = "data/models"
    os.makedirs(out, exist_ok=True)
    joblib.dump(fm, f"{out}/calibrated_model_cv5.pkl")
    joblib.dump(tfidf, f"{out}/tfidf_vectorizer_cv5.pkl")
    with open(f"{out}/thresholds_cv5.json", "w") as f:
        json.dump({"t_legit": best_tl, "t_phish": best_tp}, f, indent=2)

    # Evaluate
    pipe_old = PhishingPipeline()
    pipe_new = PhishingPipeline(
        model_path=f"{out}/calibrated_model_cv5.pkl",
        tfidf_path=f"{out}/tfidf_vectorizer_cv5.pkl",
        thresholds_path=f"{out}/thresholds_cv5.json",
    )

    print("\n=== Holdout ===")
    for tag, pipe in [("OLD", pipe_old), ("NEW", pipe_new)]:
        fp_c=leg=unc=0
        for ch in ["email","sms","whatsapp"]:
            for img in image_files(f"data/real_holdout/final/{ch}"):
                try:
                    r=pipe.predict_image(img); v=r["verdict"]
                    if v=="phishing": fp_c+=1
                    elif v=="legitimate": leg+=1
                    else: unc+=1
                except: unc+=1
        t=fp_c+leg+unc; d=fp_c+leg
        print(f"  {tag} final: fp={fp_c} leg={leg} unc={unc} N={t} fpr={fp_c/max(1,d):.3f}")

    for tag, pipe in [("OLD", pipe_old), ("NEW", pipe_new)]:
        tp_c=fn_c=unc=0
        for img in image_files("data/smishtank_test_holdout"):
            try:
                r=pipe.predict_image(img); v=r["verdict"]
                if v=="phishing": tp_c+=1
                elif v=="legitimate": fn_c+=1
                else: unc+=1
            except: unc+=1
        d=tp_c+fn_c
        print(f"  {tag} smish: tp={tp_c} fn={fn_c} unc={unc} rec={tp_c/max(1,d):.3f}")

    # Synth test
    test_df = syn_df[syn_df["split"]=="test"].copy()
    for tag, mdl, tv in [("OLD", joblib.load(f"{out}/calibrated_model.pkl"),
                           joblib.load(f"{out}/tfidf_vectorizer.pkl")),
                          ("NEW", fm, tfidf)]:
        Xt, yt, _, _ = build_feature_matrix(test_df, tfidf_vectorizer=tv, fit_tfidf=False)
        p = mdl.predict_proba(Xt)[:,1]
        if tag=="OLD":
            with open(f"{out}/thresholds.json") as f: th=json.load(f)
            tl_,tp_=th["t_legit"],th["t_phish"]
        else:
            tl_,tp_=best_tl,best_tp
        dec=(p<=tl_)|(p>=tp_)
        yd=yt[dec]; pd_=(p[dec]>=tp_).astype(int)
        tn=np.sum((yd==0)&(pd_==0)); fp=np.sum((yd==0)&(pd_==1))
        tpc=np.sum((yd==1)&(pd_==1)); fn=np.sum((yd==1)&(pd_==0))
        print(f"  {tag} synth: acc={(tn+tpc)/max(1,len(yd)):.3f} fpr={fp/max(1,tn+fp):.3f} "
              f"rec={tpc/max(1,tpc+fn):.3f} abst={1-np.mean(dec):.3f}")


def main():
    fix_split()
    blocked = get_blocked_hashes()
    logger.info("Blocked %d hashes", len(blocked))
    reingest(blocked)
    train_and_eval()


if __name__ == "__main__":
    main()
