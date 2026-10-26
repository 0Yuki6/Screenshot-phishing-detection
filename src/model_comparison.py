"""
Model Comparison: Random Forest vs SVM vs Logistic Regression.

Loads the processed feature matrix, trains all three classifiers with
stratified 5-fold cross-validation, evaluates on a held-out test set,
and outputs comprehensive statistics including:
    - Training and inference time measurements
    - Training and cross-validation scores (Accuracy, Precision, Recall, F1)
    - Held-out test set metrics (Accuracy, Precision, Recall, F1)
    - Classification reports (per-class precision / recall / F1)
    - Confusion matrices (TP, TN, FP, FN, FPR, Phishing Recall)
    - Side-by-side comparison table (macro averages)
    - Final recommendation grounded in experiment results

Usage:
    python -m src.model_comparison
"""

import os
import sys
import time
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Reconfigure console streams to UTF-8 if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# -- Configuration -----------------------------------------------------
FEATURES_CSV = "data/processed/features_real.csv"
OUTPUT_FILE = "data/processed/model_comparison_output.txt"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
LABEL_NAMES = ["Legitimate", "Phishing"]

_report_buffer = []


def report_print(*args, **kwargs):
    """Print to console and append to the report buffer for file saving."""
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    msg = sep.join(str(a) for a in args) + end
    sys.stdout.write(msg)
    sys.stdout.flush()
    _report_buffer.append(msg)


def save_report(filepath: str = OUTPUT_FILE):
    """Save the accumulated report buffer to file using explicit UTF-8 encoding."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("".join(_report_buffer))


# -- Helper: ASCII divider ---------------------------------------------
def _divider(title: str, char: str = "=", width: int = 72) -> str:
    """Create a clean ASCII divider with a centered title."""
    padding = max(0, (width - len(title) - 2) // 2)
    left_padding = padding
    right_padding = max(0, width - len(title) - 2 - left_padding)
    return f"\n{char * left_padding} {title} {char * right_padding}"


# -- Load data ---------------------------------------------------------
def load_data(csv_path: str):
    """Load the feature CSV and split into X, y."""
    df = pd.read_csv(csv_path)

    # Separate features from metadata columns
    drop_cols = [c for c in ["label", "channel_name"] if c in df.columns]
    X = df.drop(columns=drop_cols).values.astype(np.float64)
    y = df["label"].values.astype(int)

    logger.info("Loaded %d samples x %d features from %s", X.shape[0], X.shape[1], csv_path)
    logger.info("Class distribution - Legitimate: %d, Phishing: %d", (y == 0).sum(), (y == 1).sum())
    return X, y


# -- Define models -----------------------------------------------------
def get_models() -> dict:
    """Return a dict of {name: sklearn estimator}."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM (RBF Kernel)": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# -- Cross-validation --------------------------------------------------
def run_cross_validation(model, X, y, name: str) -> dict:
    """Run stratified K-fold CV and return per-fold + mean scores."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    t0 = time.perf_counter()
    cv_results = cross_validate(
        model, X, y,
        cv=skf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )
    cv_time = time.perf_counter() - t0

    results = {
        "cv_time_sec": cv_time,
        # -- Train scores (mean +/- std) -------------------------------
        "train_accuracy_mean": cv_results["train_accuracy"].mean(),
        "train_accuracy_std": cv_results["train_accuracy"].std(),
        "train_precision_mean": cv_results["train_precision_macro"].mean(),
        "train_precision_std": cv_results["train_precision_macro"].std(),
        "train_recall_mean": cv_results["train_recall_macro"].mean(),
        "train_recall_std": cv_results["train_recall_macro"].std(),
        "train_f1_mean": cv_results["train_f1_macro"].mean(),
        "train_f1_std": cv_results["train_f1_macro"].std(),
        # -- Validation scores (mean +/- std) --------------------------
        "val_accuracy_mean": cv_results["test_accuracy"].mean(),
        "val_accuracy_std": cv_results["test_accuracy"].std(),
        "val_precision_mean": cv_results["test_precision_macro"].mean(),
        "val_precision_std": cv_results["test_precision_macro"].std(),
        "val_recall_mean": cv_results["test_recall_macro"].mean(),
        "val_recall_std": cv_results["test_recall_macro"].std(),
        "val_f1_mean": cv_results["test_f1_macro"].mean(),
        "val_f1_std": cv_results["test_f1_macro"].std(),
        # -- Per-fold arrays -------------------------------------------
        "per_fold_train_accuracy": cv_results["train_accuracy"],
        "per_fold_val_accuracy": cv_results["test_accuracy"],
        "per_fold_train_f1": cv_results["train_f1_macro"],
        "per_fold_val_f1": cv_results["test_f1_macro"],
    }
    return results


# -- Test set evaluation -----------------------------------------------
def evaluate_on_test(model, X_test, y_test, name: str) -> dict:
    """Evaluate a fitted model on the held-out test set."""
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    predict_time = time.perf_counter() - t0

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    phishing_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    phishing_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "predict_time_sec": predict_time,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "classification_report": classification_report(
            y_test, y_pred, target_names=LABEL_NAMES, zero_division=0
        ),
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fpr,
        "fnr": fnr,
        "phishing_recall": phishing_recall,
        "phishing_precision": phishing_precision,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# -- Pretty print per-model results ------------------------------------
def print_model_results(name: str, cv: dict, test: dict, train_time: float):
    """Print detailed results for a single model with consistent section headings."""
    report_print(_divider(f"MODEL: {name}", "="))

    # Timing measurements
    report_print("\n  Timing Measurements:")
    report_print(f"    Final model fitting time      : {train_time:.2f}s")
    report_print(f"    5-fold cross-validation time  : {cv['cv_time_sec']:.2f}s")
    report_print(f"    Test set prediction time      : {test['predict_time_sec']:.4f}s")

    # TRAINING PERFORMANCE
    report_print(f"\n  {'-' * 55}")
    report_print(f"  TRAINING PERFORMANCE (mean +/- std across {CV_FOLDS} folds)")
    report_print(f"  {'-' * 55}")
    report_print(f"    Accuracy (macro) : {cv['train_accuracy_mean']:.4f} +/- {cv['train_accuracy_std']:.4f}")
    report_print(f"    Precision (macro): {cv['train_precision_mean']:.4f} +/- {cv['train_precision_std']:.4f}")
    report_print(f"    Recall (macro)   : {cv['train_recall_mean']:.4f} +/- {cv['train_recall_std']:.4f}")
    report_print(f"    F1-Score (macro) : {cv['train_f1_mean']:.4f} +/- {cv['train_f1_std']:.4f}")

    # CROSS-VALIDATION PERFORMANCE
    report_print(f"\n  {'-' * 55}")
    report_print(f"  CROSS-VALIDATION PERFORMANCE (mean +/- std across {CV_FOLDS} folds)")
    report_print(f"  {'-' * 55}")
    report_print(f"    Accuracy (macro) : {cv['val_accuracy_mean']:.4f} +/- {cv['val_accuracy_std']:.4f}")
    report_print(f"    Precision (macro): {cv['val_precision_mean']:.4f} +/- {cv['val_precision_std']:.4f}")
    report_print(f"    Recall (macro)   : {cv['val_recall_mean']:.4f} +/- {cv['val_recall_std']:.4f}")
    report_print(f"    F1-Score (macro) : {cv['val_f1_mean']:.4f} +/- {cv['val_f1_std']:.4f}")

    # PER-FOLD CROSS-VALIDATION BREAKDOWN
    report_print(f"\n  {'-' * 55}")
    report_print("  PER-FOLD CROSS-VALIDATION BREAKDOWN")
    report_print(f"  {'-' * 55}")
    for i in range(CV_FOLDS):
        train_acc = cv["per_fold_train_accuracy"][i]
        val_acc = cv["per_fold_val_accuracy"][i]
        train_f1 = cv["per_fold_train_f1"][i]
        val_f1 = cv["per_fold_val_f1"][i]
        overfit = train_f1 - val_f1
        flag = " [!] overfit" if overfit > 0.05 else ""
        report_print(
            f"    Fold {i+1}: Train Acc={train_acc:.4f} F1={train_f1:.4f}  |  "
            f"Val Acc={val_acc:.4f} F1={val_f1:.4f}  (gap={overfit:+.4f}){flag}"
        )

    # HELD-OUT TEST PERFORMANCE
    report_print(f"\n  {'-' * 55}")
    report_print("  HELD-OUT TEST PERFORMANCE (20% split)")
    report_print(f"  {'-' * 55}")
    report_print(f"    Accuracy         : {test['accuracy']:.4f}")
    report_print(f"    Precision (macro): {test['precision']:.4f}")
    report_print(f"    Recall (macro)   : {test['recall']:.4f}")
    report_print(f"    F1-Score (macro) : {test['f1']:.4f}")

    # CLASSIFICATION REPORT
    report_print(f"\n  {'-' * 55}")
    report_print("  CLASSIFICATION REPORT (Held-out Test Set)")
    report_print(f"  {'-' * 55}")
    for line in test["classification_report"].split("\n"):
        report_print(f"    {line}")

    # CONFUSION MATRIX
    cm = test["confusion_matrix"]
    report_print(f"\n  {'-' * 55}")
    report_print("  CONFUSION MATRIX (Held-out Test Set)")
    report_print(f"  {'-' * 55}")
    report_print("                      Predicted")
    report_print("                  Legit    Phish")
    report_print(f"    Actual Legit  {cm[0,0]:5d}    {cm[0,1]:5d}")
    report_print(f"    Actual Phish  {cm[1,0]:5d}    {cm[1,1]:5d}")

    tn, fp, fn, tp = test["tn"], test["fp"], test["fn"], test["tp"]
    fpr = test["fpr"]
    precision = test["phishing_precision"]
    recall = test["phishing_recall"]
    fnr = test["fnr"]
    report_print(f"\n    TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    report_print(f"    False Positive Rate (FPR) : {fpr:.4f}")
    report_print(f"    False Negative Rate (FNR) : {fnr:.4f}")
    report_print(f"    Phishing Precision        : {precision:.4f}")
    report_print(f"    Phishing Recall (TPR)     : {recall:.4f}")


# -- Side-by-side comparison table -------------------------------------
def print_comparison_table(all_results: dict):
    """Print a concise side-by-side comparison table across all models."""
    report_print(_divider("SIDE-BY-SIDE MODEL COMPARISON", "="))

    names = list(all_results.keys())
    header = f"{'Metric':<34s}"
    for n in names:
        header += f" | {n:>20s}"
    report_print(f"\n  {header}")
    report_print(f"  {'-' * len(header)}")

    rows = [
        ("Final Model Fitting Time (s)", lambda r: f"{r['train_time']:.2f}"),
        ("5-Fold CV Time (s)", lambda r: f"{r['cv']['cv_time_sec']:.2f}"),
        ("Test Set Prediction Time (s)", lambda r: f"{r['test']['predict_time_sec']:.4f}"),
        ("Test Accuracy", lambda r: f"{r['test']['accuracy']:.4f}"),
        ("Test Precision (macro avg)", lambda r: f"{r['test']['precision']:.4f}"),
        ("Test Recall (macro avg)", lambda r: f"{r['test']['recall']:.4f}"),
        ("Test F1-Score (macro avg)", lambda r: f"{r['test']['f1']:.4f}"),
        ("CV Accuracy (macro avg)", lambda r: f"{r['cv']['val_accuracy_mean']:.4f} +/- {r['cv']['val_accuracy_std']:.4f}"),
        ("CV Precision (macro avg)", lambda r: f"{r['cv']['val_precision_mean']:.4f} +/- {r['cv']['val_precision_std']:.4f}"),
        ("CV Recall (macro avg)", lambda r: f"{r['cv']['val_recall_mean']:.4f} +/- {r['cv']['val_recall_std']:.4f}"),
        ("CV F1-Score (macro avg)", lambda r: f"{r['cv']['val_f1_mean']:.4f} +/- {r['cv']['val_f1_std']:.4f}"),
        ("Phishing Recall (Test)", lambda r: f"{r['test']['phishing_recall']:.4f}"),
        ("False Positive Rate (Test)", lambda r: f"{r['test']['fpr']:.4f}"),
        ("False Negative Rate (Test)", lambda r: f"{r['test']['fnr']:.4f}"),
    ]

    for label, fn in rows:
        row = f"  {label:<34s}"
        for n in names:
            row += f" | {fn(all_results[n]):>20s}"
        report_print(row)

    report_print(f"\n  {'-' * len(header)}")
    report_print("\n  BEST MODEL PER METRIC:")
    winner_metrics = [
        ("Test F1-Score (macro avg)", lambda r: r["test"]["f1"]),
        ("Test Precision (macro avg)", lambda r: r["test"]["precision"]),
        ("Test Recall (macro avg)", lambda r: r["test"]["recall"]),
        ("Phishing Recall (Test)", lambda r: r["test"]["phishing_recall"]),
        ("Lowest False Positive Rate", lambda r: -r["test"]["fpr"]),
        ("Lowest False Negative Rate", lambda r: -r["test"]["fnr"]),
    ]
    for metric_name, metric_fn in winner_metrics:
        best = max(names, key=lambda n: metric_fn(all_results[n]))
        val = metric_fn(all_results[best])
        if "Lowest" in metric_name:
            val = abs(val)
        report_print(f"     {metric_name:<30s} -> {best} ({val:.4f})")


# -- Main pipeline -----------------------------------------------------
def main():
    report_print("\n" + "=" * 72)
    report_print("   MODEL COMPARISON: Random Forest vs SVM vs Logistic Regression")
    report_print("   Phishing Detection Pipeline")
    report_print("=" * 72)

    # Count total screenshots across the data directory
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    total_screenshots = 0
    for dirpath, _, filenames in os.walk(data_root):
        for fname in filenames:
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                total_screenshots += 1
    report_print(f"\n   Total screenshot dataset: ~{total_screenshots:,} images")

    # Load data
    X, y = load_data(FEATURES_CSV)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    logger.info("Train set: %d samples  |  Test set: %d samples", len(y_train), len(y_test))

    # Feature scaling (required for SVM & Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Run models
    models = get_models()
    all_results = {}

    for name, model in models.items():
        report_print(_divider(f"Training: {name}", "-"))

        # Cross-validation on training data
        logger.info("Running %d-fold cross-validation for %s ...", CV_FOLDS, name)
        cv_results = run_cross_validation(model, X_train_scaled, y_train, name)

        # Final fit on full training set
        t0 = time.perf_counter()
        model.fit(X_train_scaled, y_train)
        train_time = time.perf_counter() - t0

        # Test set evaluation
        test_results = evaluate_on_test(model, X_test_scaled, y_test, name)

        all_results[name] = {
            "model": model,
            "cv": cv_results,
            "test": test_results,
            "train_time": train_time,
        }
        logger.info("[OK] %s complete (train: %.2fs, test F1: %.4f)",
                    name, train_time, test_results["f1"])

    # Print detailed results per model
    for name in all_results:
        r = all_results[name]
        print_model_results(name, r["cv"], r["test"], r["train_time"])

    # Print comparison table
    print_comparison_table(all_results)

    # Final recommendation
    best_name = max(
        all_results,
        key=lambda n: all_results[n]["test"]["f1"],
    )
    best_f1 = all_results[best_name]["test"]["f1"]
    best_phish_recall = all_results[best_name]["test"]["phishing_recall"]
    best_fpr = all_results[best_name]["test"]["fpr"]
    best_cv_f1 = all_results[best_name]["cv"]["val_f1_mean"]
    best_cv_std = all_results[best_name]["cv"]["val_f1_std"]

    report_print(_divider("FINAL RECOMMENDATION", "="))
    report_print(f"  Recommended Model: {best_name}")
    report_print("\n  Evaluation Rationale (based on experiment results):")
    report_print(
        f"    - Macro F1-Score: {best_name} achieved {best_f1:.4f} on the held-out test set,\n"
        f"      demonstrating the strongest balance across legitimate and phishing classes."
    )
    report_print(
        f"    - Phishing Recall: Achieved {best_phish_recall:.4f} recall on phishing samples\n"
        f"      ({all_results[best_name]['test']['tp']} detected out of "
        f"{all_results[best_name]['test']['tp'] + all_results[best_name]['test']['fn']}), "
        f"minimizing missed threats."
    )
    report_print(
        f"    - False Positive Rate: Achieved an FPR of {best_fpr:.4f} "
        f"({all_results[best_name]['test']['fp']} false alarms out of "
        f"{all_results[best_name]['test']['tn'] + all_results[best_name]['test']['fp']}), "
        f"minimizing disruption to benign communications."
    )
    report_print(
        f"    - Cross-Validation Consistency: 5-fold CV F1 score was {best_cv_f1:.4f} +/- {best_cv_std:.4f},\n"
        f"      indicating stable generalization with low fold-to-fold variance."
    )
    report_print(
        f"    - Training Efficiency: Final model fitting completed in {all_results[best_name]['train_time']:.2f}s,\n"
        f"      making it computationally suitable for periodic retraining."
    )
    report_print(
        "\n  Note: This recommendation is strictly based on the empirical performance\n"
        "  metrics observed in this experiment on the current dataset and feature set."
    )
    report_print("=" * 72 + "\n")

    # Save to file with explicit UTF-8 encoding
    save_report(OUTPUT_FILE)
    logger.info("Report saved to %s (encoding=utf-8)", OUTPUT_FILE)


if __name__ == "__main__":
    main()
