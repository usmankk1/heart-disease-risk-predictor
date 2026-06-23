"""
train_and_eval.py

Train a Random Forest using the existing preprocessing pipeline
and evaluate it on the held-out test set. Saves model to models/random_forest_run.pkl
"""
from pathlib import Path
import sys
import joblib

# Add parent directory to path so src can be imported
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocess import load_and_preprocess
from src.train import train_random_forest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print('Loading and preprocessing data...')
X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(None, apply_smote=True, target_samples=4000, test_samples=1000)
print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')

print('Training Random Forest (this may take a few minutes)...')
rf = train_random_forest(X_train, y_train)

print('Evaluating on test set...')
y_pred = rf.predict(X_test)
# try to get positive-class probabilities
try:
    y_proba = rf.predict_proba(X_test)[:, 1]
except Exception:
    # fallback to decision_function scaling
    try:
        scores = rf.decision_function(X_test)
        min_s, max_s = scores.min(), scores.max()
        if np.isclose(max_s, min_s):
            y_proba = np.zeros_like(scores)
        else:
            y_proba = (scores - min_s) / (max_s - min_s)
    except Exception:
        y_proba = np.zeros(len(y_test))

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc = roc_auc_score(y_test, y_proba) if np.unique(y_proba).size > 1 else float('nan')

print(f'Random Forest Test metrics:')
print(f'  Accuracy : {acc:.4f}')
print(f'  Precision: {prec:.4f}')
print(f'  Recall   : {rec:.4f}')
print(f'  F1 Score : {f1:.4f}')
print(f'  ROC-AUC  : {roc:.4f}')

save_path = MODELS_DIR / 'random_forest_run.pkl'
joblib.dump(rf, save_path)
print(f'Model saved to {save_path}')

# save scaler and feature names if not present
scaler_path = MODELS_DIR / 'scaler.pkl'
if not scaler_path.exists():
    joblib.dump(scaler, scaler_path)
    print('Saved scaler to', scaler_path)

feat_path = MODELS_DIR / 'feature_names.txt'
if not feat_path.exists():
    with open(feat_path, 'w', encoding='utf8') as f:
        for c in feature_names:
            f.write(c + '\n')
    print('Saved feature names to', feat_path)
