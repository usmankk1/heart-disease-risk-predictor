"""
preprocess.py
─────────────
Heart Disease Risk Predictor
Handles all data loading, cleaning, encoding, scaling, and splitting.

Author: Usman Khan, Ibrahim, Abdul Hannan
SAP IDs: 37906, 24865, 39115
Course: AI 
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess(filepath):
    """
    Load the heart disease dataset and apply full preprocessing pipeline.

    Steps:
        1. Load CSV from filepath
        2. Handle impossible zero values (Cholesterol, RestingBP)
        3. Encode binary and ordinal categorical features
        4. One-hot encode nominal categorical features
        5. Drop remaining NaN rows
        6. Split into train/test sets (80/20 stratified)
        7. Apply StandardScaler (fit on train only — no data leakage)

    Args:
        filepath (str): Absolute path to heart.csv

    Returns:
        X_train_scaled (np.ndarray): Scaled training features
        X_test_scaled  (np.ndarray): Scaled test features
        y_train        (pd.Series):  Training labels
        y_test         (pd.Series):  Test labels
        scaler         (StandardScaler): Fitted scaler for inference
        feature_names  (list): Column names after encoding
    """

    df = pd.read_csv(filepath)

    # ── Handle missing-as-zero ───────────────────────────────────
    # Cholesterol = 0 is physiologically impossible (172 patients)
    # RestingBP = 0 is physiologically impossible (1 patient)
    # Strategy: replace with median computed from non-zero values
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].median())

    df['RestingBP'] = df['RestingBP'].replace(0, np.nan)
    df['RestingBP'] = df['RestingBP'].fillna(df['RestingBP'].median())

    # ── Binary encoding ──────────────────────────────────────────
    # Map M/F and Y/N to 1/0 — simple and lossless
    df['Sex']            = df['Sex'].map({'M': 1, 'F': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})

    # ── Ordinal encoding ─────────────────────────────────────────
    # ST_Slope has a natural order: Up > Flat > Down
    # Encoding preserves this ordinal relationship
    df['ST_Slope'] = df['ST_Slope'].map({'Up': 2, 'Flat': 1, 'Down': 0})

    # ── One-hot encoding ─────────────────────────────────────────
    # ChestPainType (4 categories) and RestingECG (3 categories)
    # are nominal — no natural order — so one-hot encoding is used
    # drop_first=False to retain all categories for interpretability
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG'], drop_first=False)

    # ── Drop any remaining NaN rows ──────────────────────────────
    df = df.dropna()

    # ── Separate features and target ─────────────────────────────
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    # ── Stratified train/test split ──────────────────────────────
    # stratify=y ensures both splits maintain the 55/45 class ratio
    # random_state=42 ensures reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Feature scaling ──────────────────────────────────────────
    # StandardScaler: mean=0, std=1 per feature
    # Critical for SVM (distance-based) and Logistic Regression (gradient-based)
    # IMPORTANT: fit on X_train only, transform both — prevents data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


if __name__ == '__main__':
    # Quick test — run directly to verify preprocessing works
    X_tr, X_te, y_tr, y_te, scaler, cols = load_and_preprocess('../data/heart_5000.csv')
    print(f'Train: {X_tr.shape}, Test: {X_te.shape}')
    print(f'Features ({len(cols)}): {cols}')
    print(f'Class balance — Train: {y_tr.mean():.2%} positive')