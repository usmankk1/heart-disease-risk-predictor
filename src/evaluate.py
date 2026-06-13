"""
evaluate.py
───────────
Heart Disease Risk Predictor
Contains all model evaluation functions — metrics, confusion matrix, ROC curves.

Metrics used:
    - Accuracy    : Overall correct predictions
    - Precision   : Of flagged patients, how many were truly sick
    - Recall      : Of all sick patients, how many did we catch (most critical in medical context)
    - F1-Score    : Harmonic mean of precision and recall
    - ROC-AUC     : Model's ability to rank sick patients above healthy ones at any threshold

Author: Usman Khan, Ibrahim, Abdul Hannan
SAP IDs: 37906, 24865, 39115
Course: AI
"""

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model on the test set and display results.

    Computes:
        - Full classification report (precision, recall, F1 per class)
        - ROC-AUC score
        - Confusion matrix (visualised as heatmap)

    Why recall matters most here:
        Missing a true heart disease patient (false negative) is more dangerous
        than a false alarm (false positive). Recall = TP / (TP + FN).
        A model with 91% recall catches 91 out of every 100 sick patients.

    Args:
        model      : Trained sklearn-compatible model with predict_proba()
        X_test     : Scaled test features
        y_test     : True test labels
        model_name : Display name for printing and plot titles

    Returns:
        y_pred_proba (np.ndarray): Predicted probabilities for positive class
        auc          (float):      ROC-AUC score
    """
    # Generate predictions — both class labels and probabilities
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1 (disease)

    print(f'\n── {model_name} ──')
    print(classification_report(
        y_test, y_pred,
        target_names=['No Disease', 'Heart Disease']
    ))

    # ROC-AUC: 1.0 = perfect, 0.5 = random, < 0.5 = worse than random
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f'ROC-AUC: {auc:.4f}')

    # ── Confusion matrix ────────────────────────────────────────
    # Rows = actual labels, Columns = predicted labels
    # Top-left:  TN (correctly predicted no disease)
    # Top-right: FP (false alarm — predicted disease, actually healthy)
    # Bot-left:  FN (missed! — predicted healthy, actually has disease)
    # Bot-right: TP (correctly predicted disease)
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Disease', 'Heart Disease'])
    disp.plot(cmap='Blues')
    plt.title(f'{model_name} — Confusion Matrix')
    plt.show()

    return y_pred_proba, auc


def plot_roc_curves(models_data, y_test):
    """
    Plot ROC curves for multiple models on a single figure for comparison.

    ROC (Receiver Operating Characteristic) curve plots:
        X-axis: False Positive Rate = FP / (FP + TN)  [false alarm rate]
        Y-axis: True Positive Rate  = TP / (TP + FN)  [recall / sensitivity]

    A perfect model hugs the top-left corner (AUC = 1.0).
    The diagonal dashed line represents random guessing (AUC = 0.5).
    Higher AUC = better model at distinguishing disease from no disease
    across all possible classification thresholds.

    Args:
        models_data (list): List of (model_name, y_pred_proba) tuples
        y_test      : True test labels
    """
    plt.figure(figsize=(8, 6))

    for name, y_proba in models_data:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')

    # Diagonal line = random classifier baseline
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')

    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)')
    plt.title('ROC Curve Comparison — All Models')
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.show()