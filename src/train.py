from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np

def train_logistic_regression(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    lr = LogisticRegression(max_iter=1000, random_state=42)
    grid_lr = GridSearchCV(lr, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_lr.fit(X_train, y_train)
    print(f'Best LR params: {grid_lr.best_params_}')
    print(f'Best CV F1: {grid_lr.best_score_:.4f}')
    return grid_lr.best_estimator_

def train_svm(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }
    svm = SVC(probability=True, random_state=42)
    grid_svm = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_svm.fit(X_train, y_train)
    print(f'Best SVM params: {grid_svm.best_params_}')
    print(f'Best CV F1: {grid_svm.best_score_:.4f}')
    return grid_svm.best_estimator_

def cross_validate_model(model, X_train, y_train, model_name):
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    print(f'{model_name} 10-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
    return cv_scores

from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_rf = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    print(f'Best RF params: {grid_rf.best_params_}')
    print(f'Best CV F1: {grid_rf.best_score_:.4f}')
    return grid_rf.best_estimator_

    from xgboost import XGBClassifier

def train_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    grid_xgb = GridSearchCV(xgb, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_xgb.fit(X_train, y_train)
    print(f'Best XGBoost params: {grid_xgb.best_params_}')
    print(f'Best CV F1: {grid_xgb.best_score_:.4f}')
    return grid_xgb.best_estimator_


from sklearn.neural_network import MLPClassifier

def train_mlp(X_train, y_train):
    param_grid = {
        'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [500]
    }
    mlp = MLPClassifier(random_state=42)
    grid_mlp = GridSearchCV(mlp, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_mlp.fit(X_train, y_train)
    print(f'Best MLP params: {grid_mlp.best_params_}')
    print(f'Best CV F1: {grid_mlp.best_score_:.4f}')
    return grid_mlp.best_estimator_

from sklearn.ensemble import VotingClassifier

def train_voting_classifier(lr_model, rf_optuna_model, xgb_model, X_train, y_train):
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', lr_model),
            ('rf', rf_optuna_model),
            ('xgb', xgb_model)
        ],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    print('Voting Classifier trained successfully')
    return voting_clf