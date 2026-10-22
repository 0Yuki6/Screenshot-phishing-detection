"""
Per-image confidence distribution on the 30 dev holdout images.
"""

import os
import cv2
import pandas as pd
import pytesseract
from pytesseract import Output
from src.ocr import run_dual_variant_ocr


def eval_confidence():
    dev_dir = "data/real_holdout/dev"
    records = []

    for ch in ["email", "sms", "whatsapp"]:
        p = os.path.join(dev_dir, ch)
        for f in sorted(os.listdir(p)):
            img = cv2.imread(os.path.join(p, f))
            res = run_dual_variant_ocr(img)

            # Compute min word confidence using image_to_data
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            d = pytesseract.image_to_data(clahe, output_type=Output.DATAFRAME)
            v = d[d["text"].notnull() & (d["conf"] >= 0)]
            confs = v["conf"].tolist()
            min_word_conf = min(confs) if confs else 0.0

            records.append({
                "channel": ch,
                "file": f,
                "mean_conf": res["confidence"],
                "min_word_conf": round(min_word_conf, 1),
                "word_count": res["word_count"],
                "char_len": len(res["text"]),
            })

    df_dev = pd.DataFrame(records)
    print("==========================================================================")
    print(" DEV HOLDOUT PER-IMAGE CONFIDENCE DISTRIBUTION")
    print("==========================================================================")
    for ch in ["email", "sms", "whatsapp"]:
        sub = df_dev[df_dev["channel"] == ch]
        print(f"\n--- {ch.upper()} DEV IMAGES ({len(sub)} images) ---")
        for idx, r in sub.iterrows():
            fname = r["file"][:26]
            mconf = r["mean_conf"]
            wmin = r["min_word_conf"]
            wc = r["word_count"]
            cl = r["char_len"]
            print(f"  {fname} | MeanConf: {mconf:5.1f}% | MinWordConf: {wmin:4.1f}% | Words: {wc:3d} | Chars: {cl:4d}")
        print(f"  Summary -> Mean: {sub['mean_conf'].mean():.1f}% | Lowest Mean: {sub['mean_conf'].min():.1f}% | Lowest MinWord: {sub['min_word_conf'].min():.1f}%")

    print("\n==========================================================================")
    print(" OVERALL DEV HOLDOUT SUMMARY:")
    print(f" Total Images: {len(df_dev)}")
    print(f" Overall Mean Word Confidence: {df_dev['mean_conf'].mean():.1f}%")
    print(f" Lowest Mean Image Confidence across all 30: {df_dev['mean_conf'].min():.1f}% ({df_dev.loc[df_dev['mean_conf'].idxmin(), 'file']})")
    print(f" Rejection at cutoff=35%: {(df_dev['mean_conf'] < 35.0).sum()} / {len(df_dev)} (0.0%)")
    print(f" Rejection at cutoff=40%: {(df_dev['mean_conf'] < 40.0).sum()} / {len(df_dev)} (0.0%)")
    print("==========================================================================")


if __name__ == "__main__":
    eval_confidence()
