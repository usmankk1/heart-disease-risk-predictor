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

import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np

# Ensure an outputs directory exists to store evaluation visuals
OUTPUT_DIR = r'F:\AI\HDRP\HDRP\outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model on the test set and display results.
    Saves the generated confusion matrix to the outputs directory.
    """
    # Generate predictions — both class labels and probabilities
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1 (disease)

    print(f'\n── {model_name} ──')
    print(classification_report(
        y_test, y_pred,
        target_names=['No Disease', 'Heart Disease']
    ))

    # ROC-AUC: 1.0 = perfect, 0.5 = random
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f'ROC-AUC: {auc:.4f}')

    # ── Confusion matrix ────────────────────────────────────────
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Disease', 'Heart Disease'])
    
    # Clear current figure to avoid overlap lines
    plt.figure()
    disp.plot(cmap='Blues')
    plt.title(f'{model_name} — Confusion Matrix')
    
    # Safe string formatting for filenames
    safe_filename = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    save_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{safe_filename}.png')
    
    # Save instead of blocking terminal with show()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()  # Close plot allocation memory free
    
    return y_pred_proba, auc


def plot_roc_curves(models_data, y_test):
    """
    Plot ROC curves for multiple models on a single figure for comparison.
    Saves the combined ROC comparison chart to the outputs directory.
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
    
    # Save final summary visualization
    save_path = os.path.join(OUTPUT_DIR, 'roc_curve_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'\nAll valuation figures successfully exported to: {OUTPUT_DIR}')