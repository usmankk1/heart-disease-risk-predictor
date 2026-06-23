"""
run_svm.py

Train SVM only (with RandomizedSearchCV as configured in `train.py`), save the model, and evaluate.
"""
from pathlib import Path
import joblib

from preprocess import load_and_preprocess
from train import train_svm
from evaluate import evaluate_model

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)


def main():
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(
        None, apply_smote=True, target_samples=4000, test_samples=1000)

    print('\n=== Training SVM ===')
    svm = train_svm(X_train, y_train)
    joblib.dump(svm, MODELS_DIR / 'svm_balanced.pkl')
    print('Saved SVM to', MODELS_DIR / 'svm_balanced.pkl')

    print('\n=== Evaluation ===')
    evaluate_model(svm, X_test, y_test, 'SVM Balanced')

    # persist scaler and features (overwrite with balanced versions)
    joblib.dump(scaler, MODELS_DIR / 'scaler_balanced.pkl')
    with open(MODELS_DIR / 'feature_names_balanced.txt', 'w', encoding='utf8') as f:
        for c in feature_names:
            f.write(c + '\n')

    print('\nSVM run complete.')


if __name__ == '__main__':
    main()
