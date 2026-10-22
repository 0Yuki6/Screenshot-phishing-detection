"""
Tests for the model training, evaluation, and persistence module.

Uses small synthetic data to verify the train → evaluate → save → load
cycle works correctly.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (
    train_model,
    evaluate_model,
    save_model,
    load_model,
    get_permutation_importance,
    cross_validate_model,
)


def _make_synthetic_data(n_samples=100, n_features=10, random_state=42):
    """Create simple synthetic classification data."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    # Create linearly separable data with some noise
    y = (X[:, 0] + X[:, 1] + rng.randn(n_samples) * 0.3 > 0).astype(int)
    return X, y


class TestTrainModel:
    def test_train_returns_classifier(self):
        X, y = _make_synthetic_data()
        model = train_model(X, y, n_estimators=10)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_train_predict_shape(self):
        X, y = _make_synthetic_data()
        model = train_model(X, y, n_estimators=10)
        predictions = model.predict(X)
        assert predictions.shape == (len(X),)

    def test_train_predict_proba_shape(self):
        X, y = _make_synthetic_data()
        model = train_model(X, y, n_estimators=10)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestEvaluateModel:
    def test_evaluate_returns_metrics(self):
        X, y = _make_synthetic_data()
        model = train_model(X, y, n_estimators=10)
        results = evaluate_model(model, X, y)

        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert "confusion_matrix" in results
        assert "classification_report" in results

    def test_evaluate_metrics_range(self):
        X, y = _make_synthetic_data()
        model = train_model(X, y, n_estimators=10)
        results = evaluate_model(model, X, y)

        assert 0.0 <= results["accuracy"] <= 1.0
        assert 0.0 <= results["precision"] <= 1.0
        assert 0.0 <= results["recall"] <= 1.0
        assert 0.0 <= results["f1"] <= 1.0

    def test_evaluate_with_label_names(self):
        X, y = _make_synthetic_data()
        model = train_model(X, y, n_estimators=10)
        results = evaluate_model(model, X, y, label_names=["legitimate", "phishing"])
        assert "legitimate" in results["classification_report"]
        assert "phishing" in results["classification_report"]


class TestCrossValidation:
    def test_cv_returns_arrays(self):
        X, y = _make_synthetic_data()
        stratify_key = np.array(["a_" + str(label) for label in y])
        results = cross_validate_model(X, y, stratify_key, n_splits=3, n_estimators=10)

        assert "accuracy" in results
        assert "f1" in results
        assert len(results["accuracy"]) == 3
        assert len(results["f1"]) == 3


class TestPermutationImportance:
    def test_importance_ranking(self):
        X, y = _make_synthetic_data(n_features=5)
        model = train_model(X, y, n_estimators=10)
        feature_names = [f"feat_{i}" for i in range(5)]
        importance = get_permutation_importance(
            model, X, y, feature_names, n_repeats=5,
        )

        assert len(importance) == 5
        # Each entry is (name, mean, std)
        for name, mean, std in importance:
            assert isinstance(name, str)
            assert isinstance(mean, float)
            assert isinstance(std, float)

        # Should be sorted descending
        means = [m for _, m, _ in importance]
        assert means == sorted(means, reverse=True)


class TestModelPersistence:
    def test_save_and_load(self):
        X, y = _make_synthetic_data()
        model = train_model(X, y, n_estimators=10)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            save_model(model, path)
            assert os.path.exists(path)

            loaded = load_model(path)
            original_pred = model.predict(X[:5])
            loaded_pred = loaded.predict(X[:5])
            np.testing.assert_array_equal(original_pred, loaded_pred)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
