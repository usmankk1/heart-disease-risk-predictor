"""
balanced_train.py
-----------------
Run a balanced hyperparameter search for key models and evaluate them.
Saves models to models/ folder.
"""
from pathlib import Path
import warnings
import joblib

from preprocess import load_and_preprocess
from train import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_mlp,
)
from evaluate import evaluate_model

warnings.filterwarnings('ignore')

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)


def main():
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(
        None, apply_smote=True, target_samples=4000, test_samples=1000)

    print('\n=== Training Logistic Regression ===')
    lr = train_logistic_regression(X_train, y_train)
    joblib.dump(lr, MODELS_DIR / 'logistic_balanced.pkl')

    print('\n=== Training Random Forest ===')
    rf = train_random_forest(X_train, y_train)
    joblib.dump(rf, MODELS_DIR / 'rf_balanced.pkl')

    print('\n=== Training XGBoost ===')
    xgb = train_xgboost(X_train, y_train, use_gpu=False,
                        save_path=MODELS_DIR / 'xgb_balanced.pkl')

    print('\n=== Training MLP ===')
    mlp = train_mlp(X_train, y_train)
    joblib.dump(mlp, MODELS_DIR / 'mlp_balanced.pkl')

    print('\n=== Evaluation ===')
    evaluate_model(lr, X_test, y_test, 'Logistic Balanced')
    # SVM skipped per user request
    evaluate_model(rf, X_test, y_test, 'RF Balanced')
    evaluate_model(xgb, X_test, y_test, 'XGB Balanced')
    evaluate_model(mlp, X_test, y_test, 'MLP Balanced')

    # save scaler and features
    joblib.dump(scaler, MODELS_DIR / 'scaler_balanced.pkl')
    with open(MODELS_DIR / 'feature_names_balanced.txt', 'w', encoding='utf8') as f:
        for c in feature_names:
            f.write(c + '\n')

    print('\nBalanced training complete. Models saved to models/')


if __name__ == '__main__':
    main()
