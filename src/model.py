"""
Model training, evaluation, and persistence module.

Provides functions for:
- Training a Random Forest classifier with balanced class weights
- Stratified K-fold cross-validation (stratified on channel + label)
- Model evaluation (precision, recall, F1, confusion matrix)
- Permutation feature importance
- Model save/load via joblib
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
        X_train: Training feature matrix (n_samples × n_features).
        y_train: Training labels (1D array of 0/1 or "phishing"/"legitimate").
        n_estimators: Number of trees in the forest.
        random_state: Random seed for reproducibility.

    Returns:
        Fitted RandomForestClassifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info(
        "Model trained with %d estimators on %d samples, %d features",
        n_estimators, X_train.shape[0], X_train.shape[1],
    )
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate a trained model on test data.

    Args:
        model: Fitted classifier.
        X_test: Test feature matrix.
        y_test: True test labels.
        label_names: Optional list of class label names for the report.

    Returns:
        Dictionary containing:
            - "accuracy": float
            - "precision": float (macro-averaged)
            - "recall": float (macro-averaged)
            - "f1": float (macro-averaged)
            - "confusion_matrix": np.ndarray
            - "classification_report": str
    """
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=label_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": cm,
        "classification_report": report,
    }
    logger.info("Evaluation — Accuracy: %.4f, F1: %.4f", results["accuracy"], results["f1"])
    return results


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    stratify_key: np.ndarray,
    n_splits: int = 5,
    n_estimators: int = 200,
    random_state: int = 42,
) -> Dict:
    """
    Run stratified K-fold cross-validation.

    Stratification uses a combined key (e.g. "email_phishing") to
    preserve both channel and class balance across folds.

    Args:
        X: Full feature matrix.
        y: Full label array.
        stratify_key: Array of strings for stratification (channel + "_" + label).
        n_splits: Number of CV folds.
        n_estimators: Number of trees per fold.
        random_state: Random seed.

    Returns:
        Dictionary with arrays of per-fold scores:
            - "accuracy": array of per-fold accuracy
            - "precision": array of per-fold precision
            - "recall": array of per-fold recall
            - "f1": array of per-fold F1
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    cv_results = cross_validate(
        model, X, y,
        cv=skf.split(X, stratify_key),
        scoring=scoring,
        return_train_score=False,
    )

    results = {
        "accuracy": cv_results["test_accuracy"],
        "precision": cv_results["test_precision_macro"],
        "recall": cv_results["test_recall_macro"],
        "f1": cv_results["test_f1_macro"],
    }
    logger.info(
        "CV results — F1: %.4f ± %.4f (mean ± std, %d folds)",
        results["f1"].mean(), results["f1"].std(), n_splits,
    )
    return results


def get_permutation_importance(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> List[Tuple[str, float, float]]:
    """
    Compute permutation feature importance.

    Args:
        model: Fitted classifier.
        X_test: Test feature matrix.
        y_test: True test labels.
        feature_names: List of feature names matching X_test columns.
        n_repeats: Number of permutation repeats.
        random_state: Random seed.

    Returns:
        List of (feature_name, importance_mean, importance_std) tuples,
        sorted by importance descending.
    """
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    importance_list = []
    for i, name in enumerate(feature_names):
        importance_list.append((
            name,
            result.importances_mean[i],
            result.importances_std[i],
        ))
    importance_list.sort(key=lambda x: x[1], reverse=True)
    return importance_list


def save_model(model, path: str = "data/models/model.pkl") -> None:
    """Save a fitted model to disk using joblib."""
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def load_model(path: str = "data/models/model.pkl"):
    """Load a model from disk using joblib."""
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model
