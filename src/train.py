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

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBClassifier


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression with GridSearchCV hyperparameter tuning.

    Logistic Regression finds a linear decision boundary by optimising
    the log-likelihood of the binary outcome via gradient descent.
    The sigmoid function converts raw scores to probabilities.

    Hyperparameters tuned:
        C        : Inverse regularisation strength. Smaller = stronger regularisation.
        penalty  : L1 (sparse weights) or L2 (distributed weights)
        solver   : liblinear supports both L1 and L2

    Args:
        X_train (np.ndarray): Scaled training features
        y_train (pd.Series):  Training labels

    Returns:
        Best estimator from GridSearchCV (LogisticRegression)
    """
    param_grid = {
        'C':       [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver':  ['liblinear']  # supports both l1 and l2
    }
    lr = LogisticRegression(max_iter=1000, random_state=42)

    # cv=5: 5-fold cross-validation, scoring=f1: optimise for F1 (balances precision/recall)
    grid_lr = GridSearchCV(lr, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_lr.fit(X_train, y_train)

    print(f'Best LR params: {grid_lr.best_params_}')
    print(f'Best CV F1:     {grid_lr.best_score_:.4f}')
    return grid_lr.best_estimator_


def train_svm(X_train, y_train):
    """
    Train Support Vector Machine with RBF kernel using GridSearchCV.

    SVM finds the maximum-margin hyperplane separating classes.
    RBF kernel maps data to higher dimensions to handle non-linear boundaries.
    probability=True enables predict_proba() for ROC curves and SHAP.

    Hyperparameters tuned:
        C      : Margin softness. Higher C = harder margin = more overfitting risk.
        gamma  : RBF kernel bandwidth. 'scale' = 1/(n_features * X.var())
        kernel : rbf (non-linear) or linear

    Args:
        X_train (np.ndarray): Scaled training features (scaling is critical for SVM)
        y_train (pd.Series):  Training labels

    Returns:
        Best estimator from GridSearchCV (SVC)
    """
    param_grid = {
        'C':      [0.1, 1, 10, 100],
        'gamma':  ['scale', 'auto', 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }
    svm = SVC(probability=True, random_state=42)

    grid_svm = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_svm.fit(X_train, y_train)

    print(f'Best SVM params: {grid_svm.best_params_}')
    print(f'Best CV F1:      {grid_svm.best_score_:.4f}')
    return grid_svm.best_estimator_


def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier with GridSearchCV.

    Random Forest builds many decision trees on random subsets of data (bagging)
    and aggregates their predictions. This reduces variance and overfitting
    compared to a single decision tree.

    Hyperparameters tuned:
        n_estimators     : Number of trees. More = better but slower.
        max_depth        : Maximum tree depth. None = grow until pure leaves.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf : Minimum samples required at a leaf node.

    Args:
        X_train (np.ndarray): Training features
        y_train (pd.Series):  Training labels

    Returns:
        Best estimator from GridSearchCV (RandomForestClassifier)
    """
    param_grid = {
        'n_estimators':      [100, 200, 300],
        'max_depth':         [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf':  [1, 2]
    }
    rf = RandomForestClassifier(random_state=42)

    grid_rf = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_rf.fit(X_train, y_train)

    print(f'Best RF params: {grid_rf.best_params_}')
    print(f'Best CV F1:     {grid_rf.best_score_:.4f}')
    return grid_rf.best_estimator_


def train_xgboost(X_train, y_train):
    """
    Train XGBoost classifier with GridSearchCV.

    XGBoost (Extreme Gradient Boosting) builds trees sequentially,
    each tree correcting the residual errors of the previous one.
    Uses regularisation (L1/L2) to prevent overfitting.
    eval_metric='logloss' is used for binary classification.

    Hyperparameters tuned:
        n_estimators  : Number of boosting rounds.
        max_depth     : Maximum tree depth. Deeper = more complex patterns.
        learning_rate : Step size shrinkage. Lower = slower but more precise.
        subsample     : Fraction of training samples used per tree.

    Args:
        X_train (np.ndarray): Training features
        y_train (pd.Series):  Training labels

    Returns:
        Best estimator from GridSearchCV (XGBClassifier)
    """
    param_grid = {
        'n_estimators':  [100, 200, 300],
        'max_depth':     [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample':     [0.8, 1.0]
    }
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')

    grid_xgb = GridSearchCV(xgb, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_xgb.fit(X_train, y_train)

    print(f'Best XGBoost params: {grid_xgb.best_params_}')
    print(f'Best CV F1:          {grid_xgb.best_score_:.4f}')
    return grid_xgb.best_estimator_


def train_mlp(X_train, y_train):
    """
    Train MLP (Multi-Layer Perceptron) Neural Network with GridSearchCV.

    MLP is a feedforward neural network with one or more hidden layers.
    Uses backpropagation and Adam optimiser to learn weights.
    Activation function ReLU: max(0, z) — avoids vanishing gradient problem.

    Architecture: Input(18) → Hidden → Output(1, Sigmoid)

    Hyperparameters tuned:
        hidden_layer_sizes : Number of neurons per hidden layer.
        activation         : Activation function (relu or tanh).
        alpha              : L2 regularisation term.
        max_iter           : Maximum training iterations (epochs).

    Note: MLP underperforms tree models on this dataset (83% vs 89%)
    because neural networks require larger datasets (10,000+ samples)
    to outperform well-tuned ensemble methods.

    Args:
        X_train (np.ndarray): Scaled training features (scaling critical for MLP)
        y_train (pd.Series):  Training labels

    Returns:
        Best estimator from GridSearchCV (MLPClassifier)
    """
    param_grid = {
        'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
        'activation':         ['relu', 'tanh'],
        'alpha':              [0.0001, 0.001, 0.01],
        'max_iter':           [500]
    }
    mlp = MLPClassifier(random_state=42)

    grid_mlp = GridSearchCV(mlp, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_mlp.fit(X_train, y_train)

    print(f'Best MLP params: {grid_mlp.best_params_}')
    print(f'Best CV F1:      {grid_mlp.best_score_:.4f}')
    return grid_mlp.best_estimator_

def train_voting_classifier(lr_model, rf_model, xgb_model, X_train, y_train):
    """
    Train a Voting Classifier ensemble combining LR + Grid-Optimized RF + XGBoost.

    Soft voting averages the predicted probabilities from each model
    and predicts the class with the highest average probability.
    This is more nuanced than hard voting (majority vote) because
    it weighs each model's confidence in its prediction.

    Why these three models?
        - Logistic Regression: linear boundary, fast, interpretable
        - Random Forest: best tree ensemble, handles non-linearity, newly optimized
        - XGBoost: boosting approach, different error patterns than RF

    Args:
        lr_model  : Trained Logistic Regression model
        rf_model  : GridSearch-tuned Random Forest model
        xgb_model : Trained XGBoost model
        X_train   : Training features
        y_train   : Training labels

    Returns:
        Fitted VotingClassifier
    """
    voting_clf = VotingClassifier(
        estimators=[
            ('lr',  lr_model),
            ('rf',  rf_model),
            ('xgb', xgb_model)
        ],
        voting='soft'  # use probability averaging, not majority vote
    )
    voting_clf.fit(X_train, y_train)
    print('Voting Classifier trained successfully')
    return voting_clf


def cross_validate_model(model, X_train, y_train, model_name):
    """
    Perform 10-fold cross-validation and report mean accuracy with std deviation.

    10-fold CV splits the training data into 10 equal parts,
    trains on 9 and validates on 1, rotating through all combinations.
    The mean and std give a reliable estimate of generalisation performance.

    Args:
        model      : Trained sklearn-compatible model
        X_train    : Training features
        y_train    : Training labels
        model_name : Display name for printing

    Returns:
        cv_scores (np.ndarray): Array of 10 accuracy scores
    """
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=10, scoring='accuracy'
    )
    print(f'{model_name} 10-Fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
    return cv_scores