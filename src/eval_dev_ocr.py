"""
Evaluation script for dual-variant OCR on the 30 dev holdout phone screenshots.
Analyzes status bars, dark mode, mobile contrast, and confidence distribution.
"""

import os
import cv2
import numpy as np
from src.ocr import run_dual_variant_ocr, check_unreadable_gate


def evaluate_dev_holdout():
    dev_dir = "data/real_holdout/dev"
    channels = ["email", "sms", "whatsapp"]

    summary_records = []

    for ch in channels:
        p = os.path.join(dev_dir, ch)
        files = sorted(os.listdir(p))
        print(f"\n=======================================================")
        print(f" REAL DEV HOLDOUT: {ch.upper()} ({len(files)} files)")
        print(f"=======================================================")

        ch_confs = []
        ch_wins = {"clahe": 0, "adaptive_threshold": 0}
        unreadable_count = 0

        for f in files:
            fp = os.path.join(p, f)
            img = cv2.imread(fp)
            if img is None:
                print(f"Error loading {fp}")
                continue

            # Detect dark mode based on background brightness
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray))
            is_dark_mode = mean_brightness < 110.0

            # Run dual-variant OCR
            res = run_dual_variant_ocr(img)
            conf = res["confidence"]
            winner = res["winning_variant"]
            words = res["word_count"]
            text = res["text"]

            ch_confs.append(conf)
            ch_wins[winner] += 1

            unreadable, reason = check_unreadable_gate(text, conf, min_text_len=15, min_confidence=35.0)
            if unreadable:
                unreadable_count += 1

            mode_str = "DARK" if is_dark_mode else "LIGHT"
            unr_str = f" [UNREADABLE: {reason}]" if unreadable else ""

            print(
                f"  {f[:26]} | {mode_str:5s} (B={mean_brightness:5.1f}) | "
                f"Conf: {conf:5.1f}% | Win: {winner:18s} | Words: {words:3d}{unr_str}"
            )

            summary_records.append({
                "channel": ch,
                "file": f,
                "is_dark_mode": is_dark_mode,
                "mean_brightness": mean_brightness,
                "confidence": conf,
                "winning_variant": winner,
                "word_count": words,
                "unreadable": unreadable,
                "text_snippet": text[:80],
            })

        avg_conf = sum(ch_confs) / max(1, len(ch_confs))
        print(f"-------------------------------------------------------")
        print(f"Channel {ch} Avg Conf: {avg_conf:.1f}% | Unreadable: {unreadable_count}/{len(files)}")
        print(f"Variant Wins: {ch_wins}")

    return summary_records


if __name__ == "__main__":
    evaluate_dev_holdout()
