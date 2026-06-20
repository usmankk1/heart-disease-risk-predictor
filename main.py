import os
import joblib
import warnings
warnings.filterwarnings('ignore')
from src.preprocess import load_and_preprocess
from src.train import (train_logistic_regression, train_svm,
                       train_random_forest, train_xgboost,
                       train_mlp, train_voting_classifier,
                       cross_validate_model)
from src.evaluate import evaluate_model, plot_roc_curves

DATA_PATH = r'F:\AI\HDRP\HDRP\data\heart_5000.csv'
MODEL_DIR = r'F:\AI\HDRP\HDRP\models'

def main():
    # 1. Load and preprocess
    print('Loading and preprocessing data...')
    X_train, X_test, y_train, y_test, scaler, feature_names = \
        load_and_preprocess(DATA_PATH)
    print(f'Train: {X_train.shape}, Test: {X_test.shape}')

    # 2. Train all base models
    print('\n=== Training Logistic Regression ===')
    lr_model = train_logistic_regression(X_train, y_train)

    print('\n=== Training SVM ===')
    svm_model = train_svm(X_train, y_train)  

    print('\n=== Training Random Forest ===')
    rf_model = train_random_forest(X_train, y_train)

    print('\n=== Training XGBoost ===')
    xgb_model = train_xgboost(X_train, y_train)

    print('\n=== Training MLP Neural Network ===')
    mlp_model = train_mlp(X_train, y_train)

    # 3. Train Voting Classifier (Now using the fresh, clean grid-optimized rf_model)
    print('\n=== Training Voting Classifier ===')
    voting_model = train_voting_classifier(lr_model, rf_model, xgb_model, X_train, y_train)

    # 4. Cross-validate
    print('\n=== Cross Validation ===')
    cross_validate_model(lr_model, X_train, y_train, 'Logistic Regression')
    cross_validate_model(svm_model, X_train, y_train, 'SVM')
    cross_validate_model(rf_model, X_train, y_train, 'Random Forest')
    cross_validate_model(xgb_model, X_train, y_train, 'XGBoost')
    cross_validate_model(mlp_model, X_train, y_train, 'MLP Neural Network')
    cross_validate_model(voting_model, X_train, y_train, 'Voting Classifier')

    # 5. Evaluate on test set
    print('\n=== Evaluation ===')
    lr_proba, lr_auc = evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
    svm_proba, svm_auc = evaluate_model(svm_model, X_test, y_test, 'SVM')
    rf_proba, rf_auc = evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    xgb_proba, xgb_auc = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
    mlp_proba, mlp_auc = evaluate_model(mlp_model, X_test, y_test, 'MLP Neural Network')
    voting_proba, voting_auc = evaluate_model(voting_model, X_test, y_test, 'Voting Classifier')

    # 6. ROC curves (Removed old Optuna line to keep things clean for your presentation)
    plot_roc_curves([
        ('Logistic Regression', lr_proba),
        ('SVM (RBF)', svm_proba),
        ('Random Forest', rf_proba),
        ('XGBoost', xgb_proba),
        ('MLP Neural Network', mlp_proba),
        ('Voting Classifier', voting_proba)
    ], y_test)

    # 7. Save all models to explicit folder locations
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(lr_model, os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
    joblib.dump(svm_model, os.path.join(MODEL_DIR, 'svm_rbf.pkl'))
    joblib.dump(rf_model, os.path.join(MODEL_DIR, 'random_forest.pkl'))
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgboost.pkl'))
    joblib.dump(mlp_model, os.path.join(MODEL_DIR, 'mlp.pkl'))
    joblib.dump(voting_model, os.path.join(MODEL_DIR, 'voting_classifier.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    print(f'\nAll models successfully saved to: {MODEL_DIR}')

if __name__ == '__main__':
    main()