"""
preprocess.py
─────────────
Heart Disease Risk Predictor
Handles all data loading, cleaning, encoding, scaling, and splitting.
SMOTE is applied AFTER train/test split to prevent data leakage.

Author: Usman Khan, Ibrahim, Abdul Hannan
SAP IDs: 37906, 24865, 39115
Course: AI
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_and_preprocess(filepath=None, apply_smote=True, target_samples=4000, test_samples=1000):
    """
    Load the heart disease dataset and apply full preprocessing pipeline.

    Steps:
        1. Load CSV from filepath
        2. Handle impossible zero values (Cholesterol, RestingBP)
        3. Encode binary and ordinal categorical features
        4. One-hot encode nominal categorical features
        5. Drop remaining NaN rows
        6. Split into train/test sets (80/20 stratified)
        7. Apply SMOTE on training data ONLY (no data leakage)
        8. Apply StandardScaler (fit on train only - no data leakage)

    Args:
        filepath        (str | Path | None):  Path to CSV file or directory containing CSV. If
                                             None the function will try to locate common
                                             dataset files under the project's `data/` folder.
        target_samples  (int): Total number of training samples after SMOTE (default 4000)
        test_samples    (int): Number of samples to reserve for the test set (default 1000)
        apply_smote     (bool): Whether to apply SMOTE augmentation
        target_samples  (int):  Target number of training samples after SMOTE

    Returns:
        X_train_scaled (np.ndarray): Scaled training features
        X_test_scaled  (np.ndarray): Scaled test features
        y_train        (pd.Series):  Training labels
        y_test         (pd.Series):  Test labels
        scaler         (StandardScaler): Fitted scaler for inference
        feature_names  (list): Column names after encoding
    """

    # Resolve filepath: accept None, file, or directory
    tried_paths = []
    if filepath is None:
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        candidates = [PROJECT_ROOT / 'data' / 'data_5000.csv',
                      PROJECT_ROOT / 'data' / 'data.csv',
                      PROJECT_ROOT / 'data' / 'heart.csv']
        for c in candidates:
            tried_paths.append(str(c))
            if c.exists():
                filepath = c
                break
    else:
        filepath = Path(filepath)
        # If a directory was given, look for common filenames inside it
        if filepath.is_dir():
            candidates = [filepath / 'data_5000.csv',
                          filepath / 'data.csv', filepath / 'heart.csv']
            for c in candidates:
                tried_paths.append(str(c))
                if c.exists():
                    filepath = c
                    break
        else:
            tried_paths.append(str(filepath))

    if filepath is None or (isinstance(filepath, Path) and not filepath.exists()):
        raise FileNotFoundError(
            'Could not find dataset. Tried the following paths: ' +
            ', '.join(tried_paths)
        )

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise RuntimeError(f'Failed to read CSV at {filepath}: {e}')

    # -- Handle missing-as-zero --
    # Cholesterol = 0 is physiologically impossible (172 patients)
    # RestingBP   = 0 is physiologically impossible (1 patient)
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].median())

    df['RestingBP'] = df['RestingBP'].replace(0, np.nan)
    df['RestingBP'] = df['RestingBP'].fillna(df['RestingBP'].median())

    # -- Binary encoding (handle both raw string and pre-encoded numeric forms) --
    if 'Sex' in df.columns:
        if df['Sex'].dtype == object:
            df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
        else:
            df['Sex'] = df['Sex'].astype(int)

    if 'ExerciseAngina' in df.columns:
        if df['ExerciseAngina'].dtype == object:
            df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
        else:
            df['ExerciseAngina'] = df['ExerciseAngina'].astype(int)

    # -- Ordinal encoding for ST_Slope only if needed --
    if 'ST_Slope' in df.columns:
        if df['ST_Slope'].dtype == object:
            df['ST_Slope'] = df['ST_Slope'].map(
                {'Up': 2, 'Flat': 1, 'Down': 0})
        # if numeric already, leave as-is

    # -- One-hot encoding: apply only if base categorical columns exist --
    cols_to_dummy = []
    if 'ChestPainType' in df.columns:
        cols_to_dummy.append('ChestPainType')
    if 'RestingECG' in df.columns:
        cols_to_dummy.append('RestingECG')
    if cols_to_dummy:
        df = pd.get_dummies(df, columns=cols_to_dummy, drop_first=False)

    # -- Drop any remaining NaN rows --
    df = df.dropna()

    # -- Separate features and target --
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    feature_names = X.columns.tolist()

    # -- Stratified train/test split --
    # IMPORTANT: Split BEFORE SMOTE to prevent data leakage
    # Allow specifying absolute number of test samples
    n_rows = len(X)
    if isinstance(test_samples, int) and 0 < test_samples < n_rows:
        test_size_param = test_samples
    else:
        # fallback to 20% if invalid
        test_size_param = 0.2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_param, random_state=42, stratify=y
    )

    # -- SMOTE: Synthetic Minority Oversampling --
    # Applied ONLY on training data AFTER split
    # Test set remains 100% real clinical data - no leakage
    if apply_smote:
        # target_samples is total desired training samples after SMOTE
        samples_per_class = max(target_samples // 2,
                                max(y_train.value_counts()))

        smote = SMOTE(random_state=42, k_neighbors=5,
                      sampling_strategy={0: samples_per_class, 1: samples_per_class})
        X_train_arr, y_train_arr = smote.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train_arr, columns=feature_names)
        y_train = pd.Series(y_train_arr, name='HeartDisease')

        print(
            f'SMOTE applied: Training set expanded to {len(X_train)} samples')
        print(f'Class balance after SMOTE: {y_train.value_counts().to_dict()}')
        print(f'Test set unchanged: {len(X_test)} real samples (no leakage)')

    # -- Feature scaling --
    # IMPORTANT: fit on X_train only, transform both - prevents data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


if __name__ == '__main__':
    # Resolve data path relative to the project root (one level above `src`)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_path = PROJECT_ROOT / 'data' / 'data_5000.csv'

    X_tr, X_te, y_tr, y_te, scaler, cols = load_and_preprocess(
        str(data_path), apply_smote=True, target_samples=4000, test_samples=1000
    )
    print(f'Train: {X_tr.shape}, Test: {X_te.shape}')
    print(f'Features ({len(cols)}): {cols}')
    print(f'Class balance - Train: {y_tr.mean():.2%} positive')
