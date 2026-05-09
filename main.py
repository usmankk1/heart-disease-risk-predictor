import joblib
from src.preprocess import load_and_preprocess
from src.train import train_logistic_regression, train_svm, cross_validate_model
from src.evaluate import evaluate_model, plot_roc_curves

DATA_PATH = r'F:\AI\HDRP\HDRP\data\heart.csv'

def main():
    # 1. Load and preprocess
    print('Loading and preprocessing data...')
    X_train, X_test, y_train, y_test, scaler, feature_names = \
        load_and_preprocess(DATA_PATH)
    print(f'Train: {X_train.shape}, Test: {X_test.shape}')

    # 2. Train both models
    print('\n=== Training Logistic Regression ===')
    lr_model = train_logistic_regression(X_train, y_train)

    print('\n=== Training SVM ===')
    svm_model = train_svm(X_train, y_train)

    # 3. Cross-validate
    print('\n=== Cross Validation ===')
    cross_validate_model(lr_model, X_train, y_train, 'Logistic Regression')
    cross_validate_model(svm_model, X_train, y_train, 'SVM')

    # 4. Evaluate on test set
    print('\n=== Evaluation ===')
    lr_proba, lr_auc = evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
    svm_proba, svm_auc = evaluate_model(svm_model, X_test, y_test, 'SVM')

    # 5. ROC curves
    plot_roc_curves([
        ('Logistic Regression', lr_proba),
        ('SVM (RBF)', svm_proba)
    ], y_test)

    # 6. Save models
    joblib.dump(lr_model, 'models/logistic_regression.pkl')
    joblib.dump(svm_model, 'models/svm_rbf.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print('\nModels saved to models/')

if __name__ == '__main__':
    main()
