"""
tests/test_evaluate.py
Tests for model evaluation
"""
import pytest
import numpy as np
from src.preprocess import load_and_preprocess
from src.evaluate import safe_predict_proba, _get_prediction_probabilities
from sklearn.linear_model import LogisticRegression


def test_safe_predict_proba_with_predict_proba():
    """Test safe_predict_proba with a model that supports predict_proba."""
    X_train, X_test, y_train, y_test, _, _ = load_and_preprocess(
        None, apply_smote=False, test_samples=100
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    proba = _get_prediction_probabilities(model, X_test)
    assert proba.shape[0] == X_test.shape[0]
    assert np.all((proba >= 0) & (proba <= 1))


def test_safe_predict_proba_output_shape():
    """Test that safe_predict_proba returns correct shape."""
    X_train, X_test, y_train, y_test, _, _ = load_and_preprocess(
        None, apply_smote=False, test_samples=100
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    proba = _get_prediction_probabilities(model, X_test)
    assert proba.ndim == 1
    assert len(proba) == len(y_test)
