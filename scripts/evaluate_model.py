"""
Evaluate the production Random Forest on features_real.csv
(same settings as src/model.py + train_on_real_data.py: no scaling).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import (
    train_model,
    evaluate_model,
    cross_validate_model,
    get_permutation_importance,
)

FEATURES_CSV = ROOT / "data/processed/features_real.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
LABEL_NAMES = ["Legitimate", "Phishing"]


def main():
    df = pd.read_csv(FEATURES_CSV)
    drop_cols = [c for c in ["label", "channel_name"] if c in df.columns]
    feature_names = [c for c in df.columns if c not in drop_cols]
    X = df.drop(columns=drop_cols).values.astype(np.float64)
    y = df["label"].values.astype(int)

    n_legit = int((y == 0).sum())
    n_phish = int((y == 1).sum())
    print("=" * 72)
    print("PHISHING DETECTOR — MODEL EVALUATION (Production Random Forest)")
    print("=" * 72)
    print(f"Dataset     : {FEATURES_CSV.name}")
    print(f"Samples     : {len(y)}  (Legitimate: {n_legit}, Phishing: {n_phish})")
    print(f"Features    : {X.shape[1]}  (hand-crafted + TF-IDF)")
    print(f"Classifier  : RandomForestClassifier(n_estimators=200, class_weight=balanced)")
    print(f"Preprocessing: none (no StandardScaler — matches deployed training)")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train/test  : {len(y_train)} / {len(y_test)}  (stratified, test_size={TEST_SIZE})")
    print()

    # ── 5-fold CV on full dataset (label-stratified) ──
    from sklearn.ensemble import RandomForestClassifier

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_cv = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    cv = cross_validate(
        rf_cv,
        X,
        y,
        cv=skf,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        n_jobs=-1,
    )
    print("-" * 72)
    print("5-FOLD CROSS-VALIDATION (full dataset, stratified on label)")
    print("-" * 72)
    for key, label in [
        ("test_accuracy", "Accuracy"),
        ("test_precision_macro", "Precision (macro)"),
        ("test_recall_macro", "Recall (macro)"),
        ("test_f1_macro", "F1 (macro)"),
    ]:
        m, s = cv[key].mean(), cv[key].std()
        print(f"  {label:22s}: {m:.4f}  +/-  {s:.4f}")
    print()

    # Channel-stratified CV if channel_name present
    if "channel_name" in df.columns:
        strat_key = df["channel_name"].astype(str) + "_" + df["label"].astype(str)
        ch_cv = cross_validate_model(X, y, strat_key.values, n_splits=5)
        print("-" * 72)
        print("5-FOLD CV (stratified on channel + label — matches src/model.py helper)")
        print("-" * 72)
        print(
            f"  F1 (macro)            : {ch_cv['f1'].mean():.4f}  +/-  {ch_cv['f1'].std():.4f}"
        )
        print(
            f"  Accuracy              : {ch_cv['accuracy'].mean():.4f}  +/-  {ch_cv['accuracy'].std():.4f}"
        )
        print()

    # ── Held-out test set ──
    model = train_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test, label_names=LABEL_NAMES)
    cm = results["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()

    print("-" * 72)
    print("HELD-OUT TEST SET (20%)")
    print("-" * 72)
    print(f"  Accuracy              : {results['accuracy']:.4f}")
    print(f"  Precision (macro)     : {results['precision']:.4f}")
    print(f"  Recall (macro)        : {results['recall']:.4f}")
    print(f"  F1 (macro)            : {results['f1']:.4f}")
    print()
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(f"                    Legit   Phish")
    print(f"    Actual Legit    {tn:5d}   {fp:5d}")
    print(f"    Actual Phish    {fn:5d}   {tp:5d}")
    print()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    phish_prec = tp / (tp + fp) if (tp + fp) else 0.0
    phish_rec = tp / (tp + fn) if (tp + fn) else 0.0
    legit_prec = tn / (tn + fn) if (tn + fn) else 0.0
    legit_rec = tn / (tn + fp) if (tn + fp) else 0.0
    print(f"  Phishing — precision: {phish_prec:.4f}, recall: {phish_rec:.4f}")
    print(f"  Legitimate — precision: {legit_prec:.4f}, recall: {legit_rec:.4f}")
    print(f"  False positive rate (legit -> phish): {fpr:.4f}  ({fp} / {fp + tn})")
    print(f"  False negative rate (phish -> legit): {fnr:.4f}  ({fn} / {fn + tp})")
    print()
    print("  Classification report:")
    for line in results["classification_report"].splitlines():
        print(f"    {line}")
    print()

    print("-" * 72)
    print("TOP 20 FEATURES — PERMUTATION IMPORTANCE (test set)")
    print("-" * 72)
    imp = get_permutation_importance(model, X_test, y_test, feature_names, n_repeats=10)
    for name, mean, std in imp[:20]:
        print(f"  {name:42s}  {mean:+.5f}  +/- {std:.5f}")
    print()

    out = ROOT / "data/processed/evaluation_report.json"
    report = {
        "dataset": str(FEATURES_CSV),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "cv_5fold_label_stratified": {
            "accuracy_mean": float(cv["test_accuracy"].mean()),
            "accuracy_std": float(cv["test_accuracy"].std()),
            "f1_macro_mean": float(cv["test_f1_macro"].mean()),
            "f1_macro_std": float(cv["test_f1_macro"].std()),
        },
        "held_out_test": {
            "accuracy": float(results["accuracy"]),
            "precision_macro": float(results["precision"]),
            "recall_macro": float(results["recall"]),
            "f1_macro": float(results["f1"]),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
            "phishing_precision": float(phish_prec),
            "phishing_recall": float(phish_rec),
        },
        "top_features_permutation": [
            {"name": n, "importance_mean": float(m), "importance_std": float(s)}
            for n, m, s in imp[:20]
        ],
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved JSON report: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
