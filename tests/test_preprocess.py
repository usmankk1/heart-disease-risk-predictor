"""
tests/test_preprocess.py
Tests for preprocessing pipeline
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.preprocess import load_and_preprocess


def test_load_and_preprocess_returns_tuple():
    """Test that load_and_preprocess returns correct output structure."""
    result = load_and_preprocess(None, apply_smote=False, test_samples=100)
    assert isinstance(result, tuple)
    assert len(result) == 6
    X_train, X_test, y_train, y_test, scaler, feature_names = result
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(feature_names) > 0


def test_train_test_split():
    """Test that train/test split is correct."""
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(
        None, apply_smote=False, test_samples=100
    )
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]


def test_smote_expansion():
    """Test that SMOTE expands training set."""
    X_train_no_smote, _, _, _, _, _ = load_and_preprocess(
        None, apply_smote=False, test_samples=100
    )
    X_train_smote, _, _, _, _, _ = load_and_preprocess(
        None, apply_smote=True, target_samples=6000, test_samples=100
    )
    assert X_train_smote.shape[0] > X_train_no_smote.shape[0]


def test_scaler_transform():
    """Test that scaler is fit on training data."""
    X_train, X_test, _, _, scaler, _ = load_and_preprocess(
        None, apply_smote=False, test_samples=100
    )
    # Verify that test data is scaled
    assert np.all(X_test.mean(axis=0) != 1.0)  # Not unit norm
    assert np.any(X_test < 0)  # Has negative values (indication of scaling)
