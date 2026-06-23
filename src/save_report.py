"""
save_report.py

Generates a short report and saves confusion matrix images and ROC comparison
into the `models/` directory for the balanced training run.
"""
from pathlib import Path
import joblib
import json
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from evaluate import _get_prediction_probabilities
from preprocess import load_and_preprocess

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

MODEL_FILES = {
    'Logistic': MODELS_DIR / 'logistic_balanced.pkl',
    'RF':       MODELS_DIR / 'rf_balanced.pkl',
    'XGB':      MODELS_DIR / 'xgb_balanced.pkl',
    'MLP':      MODELS_DIR / 'mlp_balanced.pkl',
}


def save_confusion_plot(y_true, y_pred, labels, out_path, title=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap='Blues', ax=ax, colorbar=True)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(
        None, apply_smote=True, target_samples=4000, test_samples=1000)

    report_lines = []
    roc_data = []
    md_rows = []

    for name, path in MODEL_FILES.items():
        if not path.exists():
            report_lines.append(f'{name}: model file not found at {path}\n')
            continue
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        y_proba = _get_prediction_probabilities(model, X_test)

        cr = classification_report(y_test, y_pred, target_names=[
                                   'No Disease', 'Heart Disease'])
        auc = roc_auc_score(y_test, y_proba) if np.unique(
            y_proba).size > 1 else float('nan')
        acc = accuracy_score(y_test, y_pred)

        # save confusion matrix image
        img_path = MODELS_DIR / f'{name.lower()}_confusion.png'
        save_confusion_plot(y_test, y_pred, [
                            'No Disease', 'Heart Disease'], img_path, title=f'{name} - Confusion Matrix')

        # collect ROC data
        roc_data.append((name, y_proba))

        # write model params (a subset)
        params = {}
        try:
            params = model.get_params()
        except Exception:
            params = {'repr': repr(model)[:200]}

        report_lines.append(f'=== {name} ===')
        report_lines.append(f'ROC-AUC: {auc:.4f}')
        report_lines.append(f'Accuracy: {acc:.4f}')
        report_lines.append('Classification Report:')
        report_lines.append(cr)
        report_lines.append('Model params (selected):')
        # dump only some keys to keep report short
        short_params = {k: params[k] for k in list(params)[:10]}
        report_lines.append(json.dumps(short_params, indent=2))
        report_lines.append('\n')
        md_rows.append({'name': name, 'accuracy': acc,
                       'roc_auc': auc, 'img': img_path.name})

    # save combined ROC plot
    plt.figure(figsize=(8, 6))
    for name, y_proba in roc_data:
        if np.unique(y_proba).size <= 1:
            print(f'Skipping ROC curve for {name}: constant prediction scores')
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Comparison')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_path = MODELS_DIR / 'roc_comparison.png'
    plt.savefig(roc_path)
    plt.close()

    # write text report
    report_path = MODELS_DIR / 'balanced_report.txt'
    with open(report_path, 'w', encoding='utf8') as f:
        f.write('Balanced Training Report\n')
        f.write('========================\n\n')
        for line in report_lines:
            f.write(line + '\n')

    # write markdown report
    md_path = MODELS_DIR / 'balanced_report.md'
    with open(md_path, 'w', encoding='utf8') as f:
        f.write('# Balanced Training Report\n\n')
        f.write(
            'Summary of models trained and evaluation on the 1,000-sample test set.\n\n')
        f.write('| Model | Accuracy | ROC-AUC | Confusion Matrix |\n')
        f.write('|---|---:|---:|:---:|\n')
        for r in md_rows:
            f.write(
                f"| {r['name']} | {r['accuracy']:.4f} | {r['roc_auc']:.4f} | ![]({r['img']}) |\n")
        f.write('\n\n')
        f.write('## Detailed Classification Reports\n\n')
        f.write('''\n```
''')
        for line in report_lines:
            f.write(line + '\n')
        f.write('''```
\n''')

    print(f'Report and images saved to {MODELS_DIR}')


if __name__ == '__main__':
    main()
