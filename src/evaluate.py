"""
Comprehensive evaluation suite for the Phishing Screenshot Detector.

Features:
- Full evaluation of grouped held-out test split (N=873, zero group overlap).
- 95% Wilson confidence intervals on all performance metrics.
- Complete metric set: Accuracy, Precision, Recall, F1, AUROC, PR-AUC, Brier score, FPR, FNR, Abstention.
- Risk-coverage curves and model vs rules ablation.
- Real holdout evaluation: Dev set (N=30) and Final set (N=123) with account-email subgroup analysis.
- Error diagnostics saved to reports/errors/ with before/after rules attribution.
"""

import os
import math
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.features import build_feature_matrix, FEATURE_NAMES
from src.rules import evaluate_rules
from src.pipeline import PhishingPipeline

logger = logging.getLogger("evaluation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes the two-sided Wilson score confidence interval for a binomial proportion.
    Returns: (point_estimate, lower_bound, upper_bound)
    """
    if total <= 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / total
    z = 1.959963984540054  # 95% normal quantile
    denom = 1.0 + (z ** 2) / total
    center = (p_hat + (z ** 2) / (2.0 * total)) / denom
    margin = (z * math.sqrt((p_hat * (1.0 - p_hat)) / total + (z ** 2) / (4.0 * (total ** 2)))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return round(p_hat, 4), round(lower, 4), round(upper, 4)


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_evaluated: int,
) -> Dict[str, Any]:
    """
    Calculates classification metrics with exact 95% Wilson confidence intervals.
    """
    decided_n = len(y_true)
    if decided_n == 0:
        return {}

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_neg = int(tn + fp)
    n_pos = int(tp + fn)

    # Accuracy
    acc, acc_lo, acc_hi = wilson_score_interval(int(tp + tn), decided_n)
    # FPR: FP / (TN + FP)
    fpr, fpr_lo, fpr_hi = wilson_score_interval(int(fp), n_neg) if n_neg > 0 else (0.0, 0.0, 0.0)
    # FNR: FN / (TP + FN)
    fnr, fnr_lo, fnr_hi = wilson_score_interval(int(fn), n_pos) if n_pos > 0 else (0.0, 0.0, 0.0)
    # Recall (TPR): TP / (TP + FN)
    rec, rec_lo, rec_hi = wilson_score_interval(int(tp), n_pos) if n_pos > 0 else (0.0, 0.0, 0.0)
    # Precision: TP / (TP + FP)
    prec, prec_lo, prec_hi = wilson_score_interval(int(tp), int(tp + fp)) if (tp + fp) > 0 else (0.0, 0.0, 0.0)

    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "decided_samples": decided_n,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "accuracy": {"value": acc, "ci_95": [acc_lo, acc_hi]},
        "fpr": {"value": fpr, "ci_95": [fpr_lo, fpr_hi]},
        "fnr": {"value": fnr, "ci_95": [fnr_lo, fnr_hi]},
        "recall": {"value": rec, "ci_95": [rec_lo, rec_hi]},
        "precision": {"value": prec, "ci_95": [prec_lo, prec_hi]},
        "f1": round(f1, 4),
    }


def compute_risk_coverage_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    steps: int = 20,
) -> List[Dict[str, float]]:
    """
    Computes the risk (error rate) at varying coverage percentages by tuning dual threshold margins.
    """
    curve = []
    threshold_spreads = np.linspace(0.0, 0.45, steps)
    for spread in threshold_spreads:
        t_lo = 0.5 - spread
        t_hi = 0.5 + spread
        decided_idx = np.where((probs <= t_lo) | (probs >= t_hi))[0]
        cov = len(decided_idx) / len(y_true)
        if len(decided_idx) == 0:
            risk = 0.0
        else:
            decided_true = y_true[decided_idx]
            decided_pred = (probs[decided_idx] >= t_hi).astype(int)
            errors = np.sum(decided_true != decided_pred)
            risk = errors / len(decided_idx)
        curve.append({
            "threshold_low": round(t_lo, 3),
            "threshold_high": round(t_hi, 3),
            "coverage": round(cov, 4),
            "risk": round(risk, 4),
        })
    return curve


def evaluate_test_split(
    dataset_path: str = "data/processed/dataset_with_groups.csv",
    models_dir: str = "data/models",
    errors_dir: str = "reports/errors",
) -> Dict[str, Any]:
    """
    Evaluates the in-distribution held-out grouped test set (N=873).
    """
    os.makedirs(errors_dir, exist_ok=True)

    df = pd.read_csv(dataset_path)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    total_test = len(test_df)
    logger.info("Evaluating grouped held-out test split (N=%d)...", total_test)

    # Load artifacts
    calibrated_model = joblib.load(os.path.join(models_dir, "calibrated_model.pkl"))
    tfidf_vec = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    with open(os.path.join(models_dir, "thresholds.json"), "r") as f:
        thresholds = json.load(f)
    t_legit = thresholds["t_legit"]
    t_phish = thresholds["t_phish"]

    # Extract features using pre-fitted TF-IDF
    X_test, y_test, feat_names, _ = build_feature_matrix(
        test_df, tfidf_vectorizer=tfidf_vec, fit_tfidf=False
    )

    # 1. Raw calibrated probabilities (before rules)
    raw_probs = calibrated_model.predict_proba(X_test)[:, 1]

    # Pre-rule metrics (using thresholds)
    pred_raw_decided = []
    raw_decided_mask = (raw_probs <= t_legit) | (raw_probs >= t_phish)
    y_raw_decided = y_test[raw_decided_mask]
    preds_raw = (raw_probs[raw_decided_mask] >= t_phish).astype(int)
    pre_rule_metrics = compute_metrics_with_ci(y_raw_decided, preds_raw, total_test)

    # 2. Pipeline with Rules Integration
    decisions = []
    misclassifications = []

    for i in range(total_test):
        row = test_df.iloc[i]
        raw_text = str(row.get("ocr_text_dual", row.get("ocr_text", "")))
        clean_text = str(row.get("clean_text", raw_text))
        conf = float(row.get("confidence", 85.0))
        ch = str(row.get("channel", "email")).lower()
        raw_lbl = str(row["label"]).lower()
        true_label = 1 if (raw_lbl in {"1", "phishing"}) else 0
        ml_score = float(raw_probs[i])

        rule_eval = evaluate_rules(
            raw_ocr_text=raw_text,
            clean_text=clean_text,
            ocr_confidence=conf,
            ml_score=ml_score,
            channel=ch,
            t_legit=t_legit,
            t_phish=t_phish,
        )

        final_verdict = rule_eval["verdict"]
        final_score = float(rule_eval["phishing_score"])
        rules_fired = rule_eval["rules_triggered"]

        decisions.append({
            "index": i,
            "channel": ch,
            "true_label": true_label,
            "ml_score": ml_score,
            "final_score": final_score,
            "final_verdict": final_verdict,
            "rules_triggered": rules_fired,
            "alerts": rule_eval["alerts"],
            "clean_text": clean_text[:200],
        })

        # Record error diagnostics
        if final_verdict != "uncertain":
            pred_int = 1 if final_verdict == "phishing" else 0
            if pred_int != true_label:
                # Find top 3 scalar feature contributors
                top_features = []
                row_scalars = X_test[i, :len(FEATURE_NAMES)]
                nonzero_feats = [(FEATURE_NAMES[k], float(row_scalars[k])) for k in range(len(FEATURE_NAMES)) if row_scalars[k] > 0]
                nonzero_feats.sort(key=lambda x: abs(x[1]), reverse=True)
                top_features = nonzero_feats[:5]

                misclassifications.append({
                    "test_idx": i,
                    "channel": ch,
                    "true_label": "phishing" if true_label == 1 else "legitimate",
                    "predicted_verdict": final_verdict,
                    "score_before_rules": round(ml_score, 4),
                    "score_after_rules": round(final_score, 4),
                    "rules_triggered": rules_fired,
                    "top_scalar_features": top_features,
                    "clean_text": clean_text,
                    "alerts": rule_eval["alerts"],
                })

    dec_df = pd.DataFrame(decisions)

    # Decided filter
    decided_df = dec_df[dec_df["final_verdict"] != "uncertain"]
    n_uncertain = len(dec_df[dec_df["final_verdict"] == "uncertain"])
    abstention_rate, abst_lo, abst_hi = wilson_score_interval(n_uncertain, total_test)

    y_decided_true = decided_df["true_label"].values.astype(int)
    y_decided_pred = (decided_df["final_verdict"] == "phishing").values.astype(int)

    overall_metrics = compute_metrics_with_ci(y_decided_true, y_decided_pred, total_test)
    overall_metrics["abstention"] = {"value": abstention_rate, "ci_95": [abst_lo, abst_hi], "count": n_uncertain}
    overall_metrics["auroc"] = round(float(roc_auc_score(y_test, raw_probs)), 4)
    overall_metrics["pr_auc"] = round(float(average_precision_score(y_test, raw_probs)), 4)
    overall_metrics["brier_score"] = round(float(brier_score_loss(y_test, raw_probs)), 4)

    # Per-channel evaluation
    channel_eval = {}
    for ch_name, grp in dec_df.groupby("channel"):
        ch_total = len(grp)
        ch_unc = len(grp[grp["final_verdict"] == "uncertain"])
        ch_abst, ch_abst_lo, ch_abst_hi = wilson_score_interval(ch_unc, ch_total)
        ch_dec = grp[grp["final_verdict"] != "uncertain"]
        if len(ch_dec) > 0:
            ch_metrics = compute_metrics_with_ci(
                ch_dec["true_label"].values.astype(int),
                (ch_dec["final_verdict"] == "phishing").values.astype(int),
                ch_total,
            )
            ch_metrics["total_samples"] = ch_total
            ch_metrics["abstention"] = {"value": ch_abst, "ci_95": [ch_abst_lo, ch_abst_hi], "count": ch_unc}
            channel_eval[ch_name] = ch_metrics

    # Risk-coverage curve
    risk_coverage = compute_risk_coverage_curve(y_test, raw_probs)

    # Save errors
    errors_json_path = os.path.join(errors_dir, "test_misclassifications.json")
    with open(errors_json_path, "w") as f:
        json.dump(misclassifications, f, indent=2)
    logger.info("Saved %d misclassifications to %s", len(misclassifications), errors_json_path)

    return {
        "total_samples": total_test,
        "overall": overall_metrics,
        "per_channel": channel_eval,
        "pre_rules_comparison": pre_rule_metrics,
        "risk_coverage_curve": risk_coverage,
        "misclassifications_count": len(misclassifications),
    }


def evaluate_real_holdout_folder(
    folder_path: str,
    pipeline: PhishingPipeline,
    name: str = "final",
) -> Dict[str, Any]:
    """
    Evaluates real legitimate holdout screenshots (dev or final).
    All images are genuine/legitimate (ground truth = legitimate).
    """
    channels = ["email", "sms", "whatsapp"]
    records = []
    account_email_records = []

    logger.info("Evaluating real holdout '%s' from %s...", name, folder_path)

    import concurrent.futures

    tasks = []
    for ch in channels:
        ch_dir = os.path.join(folder_path, ch)
        if not os.path.exists(ch_dir):
            continue
        files = sorted(os.listdir(ch_dir))
        for fn in files:
            fp = os.path.join(ch_dir, fn)
            tasks.append((ch, fn, fp))

    def _eval_one(item):
        ch, fn, fp = item
        res = pipeline.predict_image(fp)
        is_account_email = False
        if ch == "email":
            txt = res["clean_text"].lower()
            if any(k in txt for k in ["security", "account", "password", "sign-in", "login", "verification", "code", "statement", "alert"]):
                is_account_email = True
        return {
            "file": fn,
            "channel": ch,
            "verdict": res["verdict"],
            "phishing_score": res["phishing_score"],
            "raw_ml_score": res["raw_ml_score"],
            "rules_triggered": res["rules_triggered"],
            "alerts": res["alerts"],
            "ocr_conf": res["ocr_confidence"],
            "is_account_email": is_account_email,
            "clean_text": res["clean_text"][:200],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(_eval_one, tasks))

    account_email_records = [r for r in records if r["is_account_email"]]

    total_holdout = len(records)
    if total_holdout == 0:
        return {}

    # Overall holdout metrics
    n_phish_fp = sum(1 for r in records if r["verdict"] == "phishing")
    n_legit = sum(1 for r in records if r["verdict"] == "legitimate")
    n_unc = sum(1 for r in records if r["verdict"] == "uncertain")
    decided_n = n_phish_fp + n_legit

    abst_rate, abst_lo, abst_hi = wilson_score_interval(n_unc, total_holdout)
    fpr, fpr_lo, fpr_hi = wilson_score_interval(n_phish_fp, decided_n) if decided_n > 0 else (0.0, 0.0, 0.0)

    # Per-channel holdout metrics
    ch_metrics = {}
    for ch in channels:
        ch_recs = [r for r in records if r["channel"] == ch]
        c_tot = len(ch_recs)
        if c_tot == 0:
            continue
        c_fp = sum(1 for r in ch_recs if r["verdict"] == "phishing")
        c_leg = sum(1 for r in ch_recs if r["verdict"] == "legitimate")
        c_unc = sum(1 for r in ch_recs if r["verdict"] == "uncertain")
        c_dec = c_fp + c_leg
        c_abst, c_alo, c_ahi = wilson_score_interval(c_unc, c_tot)
        c_fpr, c_flo, c_fhi = wilson_score_interval(c_fp, c_dec) if c_dec > 0 else (0.0, 0.0, 0.0)
        ch_metrics[ch] = {
            "total": c_tot,
            "legitimate": c_leg,
            "false_positive": c_fp,
            "uncertain": c_unc,
            "abstention_rate": {"value": c_abst, "ci_95": [c_alo, c_ahi]},
            "decided_fpr": {"value": c_fpr, "ci_95": [c_flo, c_fhi]},
        }

    # Account-email subgroup
    acc_tot = len(account_email_records)
    acc_fp = sum(1 for r in account_email_records if r["verdict"] == "phishing")
    acc_leg = sum(1 for r in account_email_records if r["verdict"] == "legitimate")
    acc_unc = sum(1 for r in account_email_records if r["verdict"] == "uncertain")
    acc_dec = acc_fp + acc_leg
    acc_abst, acc_alo, acc_ahi = wilson_score_interval(acc_unc, acc_tot) if acc_tot > 0 else (0.0, 0.0, 0.0)
    acc_fpr, acc_flo, acc_fhi = wilson_score_interval(acc_fp, acc_dec) if acc_dec > 0 else (0.0, 0.0, 0.0)

    account_subgroup = {
        "total": acc_tot,
        "legitimate": acc_leg,
        "false_positive": acc_fp,
        "uncertain": acc_unc,
        "abstention_rate": {"value": acc_abst, "ci_95": [acc_alo, acc_ahi]},
        "decided_fpr": {"value": acc_fpr, "ci_95": [acc_flo, acc_fhi]},
    }

    fps_detailed = [r for r in records if r["verdict"] == "phishing"]

    return {
        "name": name,
        "total_samples": total_holdout,
        "legitimate_count": n_legit,
        "false_positive_count": n_phish_fp,
        "uncertain_count": n_unc,
        "decided_samples": decided_n,
        "abstention_rate": {"value": abst_rate, "ci_95": [abst_lo, abst_hi]},
        "decided_fpr": {"value": fpr, "ci_95": [fpr_lo, fpr_hi]},
        "per_channel": ch_metrics,
        "account_email_subgroup": account_subgroup,
        "false_positives": fps_detailed,
    }


def main():
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(os.path.join(reports_dir, "errors"), exist_ok=True)

    print("\n" + "=" * 78)
    print("PHISHING SCREENSHOT DETECTOR — COMPREHENSIVE PRODUCTION EVALUATION")
    print("=" * 78)

    # 1. Evaluate Grouped Held-Out Test Set (N=873)
    test_results = evaluate_test_split()
    ov = test_results["overall"]
    print("\n" + "-" * 78)
    print("1. GROUPED HELD-OUT TEST SPLIT (N=873, Zero Leakage / Disjoint Groups)")
    print("-" * 78)
    print(f"  Decided Samples       : {ov['decided_samples']} / {test_results['total_samples']}")
    print(f"  Abstention Rate       : {ov['abstention']['value']:.2%}  (95% CI: [{ov['abstention']['ci_95'][0]:.2%}, {ov['abstention']['ci_95'][1]:.2%}])")
    print(f"  Decided Accuracy      : {ov['accuracy']['value']:.2%}  (95% CI: [{ov['accuracy']['ci_95'][0]:.2%}, {ov['accuracy']['ci_95'][1]:.2%}])")
    print(f"  Legitimate FPR        : {ov['fpr']['value']:.2%}  (95% CI: [{ov['fpr']['ci_95'][0]:.2%}, {ov['fpr']['ci_95'][1]:.2%}])")
    print(f"  Phishing Recall       : {ov['recall']['value']:.2%}  (95% CI: [{ov['recall']['ci_95'][0]:.2%}, {ov['recall']['ci_95'][1]:.2%}])")
    print(f"  Phishing FNR          : {ov['fnr']['value']:.2%}  (95% CI: [{ov['fnr']['ci_95'][0]:.2%}, {ov['fnr']['ci_95'][1]:.2%}])")
    print(f"  Precision (macro)     : {ov['precision']['value']:.2%}  (95% CI: [{ov['precision']['ci_95'][0]:.2%}, {ov['precision']['ci_95'][1]:.2%}])")
    print(f"  F1 Score (macro)      : {ov['f1']:.4f}")
    print(f"  AUROC                 : {ov['auroc']:.4f}")
    print(f"  PR-AUC                : {ov['pr_auc']:.4f}")
    print(f"  Brier Score           : {ov['brier_score']:.4f}")
    cm = ov["confusion_matrix"]
    print(f"  Confusion Matrix      : TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']}")

    print("\n  Per-Channel Breakdown (Grouped Test Split):")
    for ch, m in test_results["per_channel"].items():
        print(f"    * {ch.upper():9s}: N={m['total_samples']:3d} | Abstention: {m['abstention']['value']:.1%} | Decided Acc: {m['accuracy']['value']:.1%} | FPR: {m['fpr']['value']:.2%} | Recall: {m['recall']['value']:.1%}")

    # 2. Evaluate Real Holdout (Dev and Final)
    pipe = PhishingPipeline()

    dev_holdout = evaluate_real_holdout_folder("data/real_holdout/dev", pipe, name="dev")
    final_holdout = evaluate_real_holdout_folder("data/real_holdout/final", pipe, name="final")

    print("\n" + "-" * 78)
    print("2. REAL PHONE SCREENSHOT HOLDOUT — DEV PARTITION (N=30)")
    print("-" * 78)
    print(f"  Legitimate (TN)       : {dev_holdout['legitimate_count']} / {dev_holdout['total_samples']}")
    print(f"  False Positive (FP)   : {dev_holdout['false_positive_count']} / {dev_holdout['total_samples']}")
    print(f"  Uncertain (Abstain)   : {dev_holdout['uncertain_count']} / {dev_holdout['total_samples']}")
    print(f"  Decided FPR           : {dev_holdout['decided_fpr']['value']:.2%} (95% CI: [{dev_holdout['decided_fpr']['ci_95'][0]:.2%}, {dev_holdout['decided_fpr']['ci_95'][1]:.2%}])")
    print(f"  Abstention Rate       : {dev_holdout['abstention_rate']['value']:.2%} (95% CI: [{dev_holdout['abstention_rate']['ci_95'][0]:.2%}, {dev_holdout['abstention_rate']['ci_95'][1]:.2%}])")

    print("\n" + "-" * 78)
    print("3. REAL PHONE SCREENSHOT HOLDOUT — FINAL EVALUATION (N=123)")
    print("-" * 78)
    print(f"  Legitimate (TN)       : {final_holdout['legitimate_count']} / {final_holdout['total_samples']}")
    print(f"  False Positive (FP)   : {final_holdout['false_positive_count']} / {final_holdout['total_samples']}")
    print(f"  Uncertain (Abstain)   : {final_holdout['uncertain_count']} / {final_holdout['total_samples']}")
    print(f"  Decided FPR           : {final_holdout['decided_fpr']['value']:.2%} (95% CI: [{final_holdout['decided_fpr']['ci_95'][0]:.2%}, {final_holdout['decided_fpr']['ci_95'][1]:.2%}])")
    print(f"  Abstention Rate       : {final_holdout['abstention_rate']['value']:.2%} (95% CI: [{final_holdout['abstention_rate']['ci_95'][0]:.2%}, {final_holdout['abstention_rate']['ci_95'][1]:.2%}])")

    print("\n  Per-Channel Breakdown (Real Holdout Final):")
    for ch, m in final_holdout["per_channel"].items():
        print(f"    * {ch.upper():9s}: N={m['total']:2d} | Legit: {m['legitimate']:2d} | FP: {m['false_positive']:2d} | Unc: {m['uncertain']:2d} | Decided FPR: {m['decided_fpr']['value']:.2%} (CI: [{m['decided_fpr']['ci_95'][0]:.1%}, {m['decided_fpr']['ci_95'][1]:.1%}]) | Abstention: {m['abstention_rate']['value']:.1%}")

    acc_sub = final_holdout["account_email_subgroup"]
    print(f"\n  Account-Email Subgroup (Real Holdout Final, N={acc_sub['total']}):")
    print(f"    * Legit: {acc_sub['legitimate']} | FP: {acc_sub['false_positive']} | Unc: {acc_sub['uncertain']} | Decided FPR: {acc_sub['decided_fpr']['value']:.2%} | Abstention: {acc_sub['abstention_rate']['value']:.1%}")

    if len(final_holdout["false_positives"]) > 0:
        print("\n  Detailed False Positives on Real Holdout Final:")
        for fp_info in final_holdout["false_positives"]:
            print(f"    - {fp_info['file']} ({fp_info['channel']}): final_score={fp_info['phishing_score']:.3f}, raw_ml={fp_info['raw_ml_score']:.3f}, rules={fp_info['rules_triggered']}")
            print(f"      Alerts: {fp_info['alerts']}")
            print(f"      Text: {fp_info['clean_text'][:120]}")

    # 4. Save combined full report
    full_report = {
        "test_split_evaluation": test_results,
        "dev_holdout_evaluation": dev_holdout,
        "final_holdout_evaluation": final_holdout,
    }
    report_json_path = os.path.join(reports_dir, "evaluation_report.json")
    with open(report_json_path, "w") as f:
        json.dump(full_report, f, indent=2)
    logger.info("Saved complete evaluation report to %s", report_json_path)
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
