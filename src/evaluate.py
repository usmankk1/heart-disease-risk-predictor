"""
evaluate.py
───────────
Heart Disease Risk Predictor
Contains all model evaluation functions: metrics, confusion matrix, ROC curves.

Author: Usman Khan, Ibrahim, Abdul Hannan
SAP IDs: 37906, 24865, 39115
Course: AI
"""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np


def _get_prediction_probabilities(model, X):
    """Return class probabilities or a decision score fallback."""
    if hasattr(model, 'predict_proba'):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass

    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X)
        if scores.ndim > 1 and scores.shape[1] > 1:
            scores = scores[:, 1]
        min_score, max_score = scores.min(), scores.max()
        if np.isclose(max_score, min_score):
            return np.zeros_like(scores, dtype=float)
        return (scores - min_score) / (max_score - min_score)

    return np.full(shape=X.shape[0], fill_value=0.0, dtype=float)


def safe_predict_proba(model, X):
    """Alias for compatibility with the test suite."""
    return _get_prediction_probabilities(model, X)


def evaluate_model(model, X_test, y_test, model_name, show_plot=True):
    """
    Evaluate a trained model on the real test set.
    Note: Test set contains only real clinical data (no SMOTE samples).

    Args:
        model      : Trained sklearn-compatible model
        X_test     : Scaled real test features
        y_test     : True test labels
        model_name : Display name
        show_plot  : Whether to display the confusion matrix plot

    Returns:
        y_pred_proba (np.ndarray): Predicted probabilities
        auc          (float):      ROC-AUC score
    """
    y_pred = model.predict(X_test)
    y_pred_proba = _get_prediction_probabilities(model, X_test)

    print(f'\n{"="*50}')
    print(f' {model_name}')
    print(f'{"="*50}')
    print(classification_report(
        y_test, y_pred,
        target_names=['No Disease', 'Heart Disease']
    ))

    auc = float('nan')
    if np.unique(y_pred_proba).size > 1:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f'ROC-AUC: {auc:.4f}')
    else:
        print('ROC-AUC: not available (constant prediction scores)')

    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        cm, display_labels=['No Disease', 'Heart Disease'])
    disp.plot(cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close()

    return y_pred_proba, auc


def plot_roc_curves(models_data, y_test):
    """
    Plot ROC curves for all models on a single figure.

    Args:
        models_data (list): List of (model_name, y_pred_proba) tuples
        y_test      : True test labels
    """
    plt.figure(figsize=(10, 7))
    for name, y_proba in models_data:
        if np.unique(y_proba).size <= 1:
            print(f'Skipping ROC curve for {name}: constant prediction scores')
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)')
    plt.title('ROC Curve Comparison - All 7 Models')
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.show()
