"""
quick_train.py
--------------
Train a fast logistic regression baseline and persist artifacts for inference.
"""
from src.preprocess import load_and_preprocess
from pathlib import Path
import sys
import os
import joblib
from sklearn.linear_model import LogisticRegression

# Ensure `src` package is importable when running from project root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Fast pipeline: use auto-discovery in preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(
        None, apply_smote=True, target_samples=4000, test_samples=1000)

    # Train a simple logistic regression quickly
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train, y_train)

    # Persist model, scaler, and feature names
    joblib.dump(lr, MODELS_DIR / 'logistic_baseline.pkl')
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    with open(MODELS_DIR / 'feature_names.txt', 'w', encoding='utf8') as f:
        for col in feature_names:
            f.write(col + '\n')

    print('Saved logistic_baseline.pkl, scaler.pkl, feature_names.txt')


if __name__ == '__main__':
    main()
