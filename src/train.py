"""
train.py
────────
Heart Disease Risk Predictor
Contains training functions for all 7 models with hyperparameter tuning.

Models:
    1. Logistic Regression  (GridSearchCV)
    2. SVM RBF              (GridSearchCV)
    3. Random Forest        (GridSearchCV)
    4. XGBoost              (GridSearchCV)
    5. MLP Neural Network   (GridSearchCV)
    6. Random Forest Optuna (Bayesian optimisation)
    7. Voting Classifier    (Ensemble of LR + RF Optuna + XGBoost)

Author: Usman Khan, Ibrahim, Abdul Hannan
SAP IDs: 37906, 24865, 39115
Course: AI
"""

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import optuna
import numpy as np
import warnings
warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_logistic_regression(X_train, y_train):
    param_grid = {
        'C':       [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver':  ['liblinear']
    }
    lr = LogisticRegression(max_iter=1000, random_state=42)
    grid_lr = RandomizedSearchCV(
        lr, param_grid, n_iter=50, cv=3, scoring='f1', n_jobs=-1, random_state=42)
    grid_lr.fit(X_train, y_train)
    print(f'Best LR params: {grid_lr.best_params_}')
    print(f'Best CV F1:     {grid_lr.best_score_:.4f}')
    return grid_lr.best_estimator_


def train_svm(X_train, y_train):
    param_grid = {
        # narrowed grid for faster search
        'C':      [0.1, 1, 10],
        'gamma':  ['scale', 0.01],
        'kernel': ['rbf']
    }
    svm = SVC(probability=True, random_state=42)
    grid_svm = RandomizedSearchCV(
        svm, param_grid, n_iter=12, cv=2, scoring='f1', n_jobs=-1, random_state=42)
    grid_svm.fit(X_train, y_train)
    print(f'Best SVM params: {grid_svm.best_params_}')
    print(f'Best CV F1:      {grid_svm.best_score_:.4f}')
    return grid_svm.best_estimator_


def train_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators':      [100, 200, 300],
        'max_depth':         [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf':  [1, 2]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_rf = RandomizedSearchCV(
        rf, param_grid, n_iter=50, cv=3, scoring='f1', n_jobs=-1, random_state=42)
    grid_rf.fit(X_train, y_train)
    print(f'Best RF params: {grid_rf.best_params_}')
    print(f'Best CV F1:     {grid_rf.best_score_:.4f}')
    return grid_rf.best_estimator_


def train_xgboost(X_train, y_train, use_gpu=False, save_path=None):
    param_grid = {
        'n_estimators':  [100, 200, 300],
        'max_depth':     [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample':     [0.8, 1.0]
    }
    # choose tree method: GPU if requested and available, otherwise fast 'hist'
    tree_method = 'gpu_hist' if use_gpu else 'hist'
    xgb = XGBClassifier(random_state=42, eval_metric='logloss',
                        use_label_encoder=False, n_jobs=-1, tree_method=tree_method)
    grid_xgb = RandomizedSearchCV(
        xgb, param_grid, n_iter=50, cv=3, scoring='f1', n_jobs=-1, random_state=42)
    used_tree_method = tree_method
    try:
        grid_xgb.fit(X_train, y_train)
    except Exception as e:
        # If GPU tree method isn't available in this XGBoost build, fall back to CPU 'hist'
        print('XGBoost training with gpu_hist failed, retrying with CPU hist. Error:', e)
        used_tree_method = 'hist'
        xgb = XGBClassifier(random_state=42, eval_metric='logloss',
                            use_label_encoder=False, n_jobs=-1, tree_method=used_tree_method)
        grid_xgb = RandomizedSearchCV(
            xgb, param_grid, n_iter=20, cv=3, scoring='f1', n_jobs=-1, random_state=42)
        grid_xgb.fit(X_train, y_train)
    print(f'Best XGBoost params: {grid_xgb.best_params_}')
    print(f'Best CV F1:          {grid_xgb.best_score_:.4f}')

    # Refit final XGBoost with early stopping on a small validation split
    best_params = grid_xgb.best_params_
    final_xgb = XGBClassifier(**best_params, random_state=42,
                              eval_metric='logloss', use_label_encoder=False, n_jobs=-1,
                              tree_method=used_tree_method)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    try:
        final_xgb.fit(X_tr, y_tr, eval_set=[
                      (X_val, y_val)], early_stopping_rounds=10, verbose=False)
    except TypeError:
        # Some xgboost/sklearn API builds may not accept early_stopping_rounds as a kwarg
        print('XGBoost .fit() does not accept early_stopping_rounds on this build; fitting without early stopping.')
        final_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    except Exception as e:
        print('Warning: XGBoost final fit with early stopping failed, falling back to simple fit. Error:', e)
        final_xgb.fit(X_train, y_train)

    # optionally save final model
    if save_path:
        import joblib
        save_p = save_path if isinstance(save_path, str) else str(save_path)
        joblib.dump(final_xgb, save_p)
        print(f'Final XGBoost model saved to {save_p}')

    return final_xgb


def train_mlp(X_train, y_train):
    param_grid = {
        'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
        'activation':         ['relu', 'tanh'],
        'alpha':              [0.0001, 0.001, 0.01],
        'max_iter':           [500]
    }
    mlp = MLPClassifier(random_state=42)
    grid_mlp = RandomizedSearchCV(
        mlp, param_grid, n_iter=20, cv=3, scoring='f1', n_jobs=-1, random_state=42)
    grid_mlp.fit(X_train, y_train)
    print(f'Best MLP params: {grid_mlp.best_params_}')
    print(f'Best CV F1:      {grid_mlp.best_score_:.4f}')
    return grid_mlp.best_estimator_


def train_random_forest_optuna(X_train, y_train, n_trials=20):
    """
    Train Random Forest with Optuna Bayesian optimisation.
    More efficient than GridSearchCV for large search spaces.
    """
    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
            'max_depth':         trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state':      42
        }
        rf = RandomForestClassifier(**params)
        scores = cross_val_score(rf, X_train, y_train,
                                 cv=5, scoring='f1', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params['random_state'] = 42
    print(f'Best Optuna params: {best_params}')
    print(f'Best Optuna F1:     {study.best_value:.4f}')

    rf_optuna = RandomForestClassifier(**best_params)
    rf_optuna.fit(X_train, y_train)
    return rf_optuna


def train_voting_classifier(lr_model, rf_optuna_model, xgb_model, X_train, y_train):
    voting_clf = VotingClassifier(
        estimators=[
            ('lr',  lr_model),
            ('rf',  rf_optuna_model),
            ('xgb', xgb_model)
        ],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    print('Voting Classifier trained successfully')
    return voting_clf


def cross_validate_model(model, X_train, y_train, model_name):
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=10, scoring='accuracy'
    )
    print(f'{model_name} 10-Fold CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')
    return cv_scores
