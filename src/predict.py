"""
predict.py
----------
Loads saved scaler, feature names and model for inference.
Provides a `predict(df)` function that returns predictions and probabilities.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
MODEL_PATH = MODELS_DIR / 'logistic_baseline.pkl'
SCALER_PATH = MODELS_DIR / 'scaler.pkl'
FEATURES_PATH = MODELS_DIR / 'feature_names.txt'

if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not FEATURES_PATH.exists():
    raise FileNotFoundError(
        'Model artifacts not found in models/. Run src/quick_train.py first.')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(FEATURES_PATH, 'r', encoding='utf8') as f:
    feature_names = [ln.strip() for ln in f if ln.strip()]


def prepare_input(df: pd.DataFrame) -> np.ndarray:
    # Ensure all required features present; missing features -> fill 0
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        # add missing columns with zeros
        for c in missing:
            df[c] = 0
    # Keep only known feature order
    X = df[feature_names].astype(float)
    X_scaled = scaler.transform(X)
    return X_scaled


def predict(df: pd.DataFrame):
    """Return (preds, probs) where preds are 0/1 and probs are positive-class probability."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame([df])
    X = prepare_input(df.copy())
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    return preds, probs


if __name__ == '__main__':
    # quick smoke test using first row of data file if available
    data_path = Path(__file__).resolve().parent.parent / \
        'data' / 'data_5000.csv'
    if data_path.exists():
        df = pd.read_csv(data_path)
        # If one-hot columns present, keep those exact columns
        sample = df.iloc[[0]]
        preds, probs = predict(sample)
        print('Pred:', preds, 'Prob:', probs)
    else:
        print('No data_5000.csv found for smoke test.')

    # CLI: accept a JSON string or CSV path
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description='Run inference using saved model')
    parser.add_argument(
        '--json', type=str, help='JSON string or path to JSON file containing a single record')
    parser.add_argument('--csv', type=str,
                        help='Path to CSV file containing one or more records')
    args = parser.parse_args()

    if args.csv:
        df_in = pd.read_csv(args.csv)
        preds, probs = predict(df_in)
        for p, pr in zip(preds, probs):
            print(p, float(pr))
        sys.exit(0)

    if args.json:
        # detect if it's a file path
        try:
            p = Path(args.json)
            if p.exists():
                payload = json.loads(p.read_text())
            else:
                payload = json.loads(args.json)
        except Exception as e:
            print('Failed to parse JSON input:', e)
            sys.exit(2)
        # payload may be dict or list
        if isinstance(payload, list):
            df_in = pd.DataFrame(payload)
        else:
            df_in = pd.DataFrame([payload])
        preds, probs = predict(df_in)
        for p, pr in zip(preds, probs):
            print(p, float(pr))
        sys.exit(0)
