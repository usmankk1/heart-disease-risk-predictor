
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f'\n── {model_name} ──')
    print(classification_report(y_test, y_pred,
          target_names=['No Disease', 'Heart Disease']))

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f'ROC-AUC: {auc:.4f}')

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Disease', 'Heart Disease'])
    disp.plot(cmap='Blues')
    plt.title(f'{model_name} — Confusion Matrix')
    plt.show()

    return y_pred_proba, auc

def plot_roc_curves(models_data, y_test):
    plt.figure(figsize=(8, 6))
    for name, y_proba in models_data:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()  