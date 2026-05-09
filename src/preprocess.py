import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # Handle missing-as-zero
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].median())

    df['RestingBP'] = df['RestingBP'].replace(0, np.nan)
    df['RestingBP'] = df['RestingBP'].fillna(df['RestingBP'].median())

    # Encode binary columns
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})

    # Encode ordinal
    df['ST_Slope'] = df['ST_Slope'].map({'Up': 2, 'Flat': 1, 'Down': 0})

    # One-hot encode
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG'], drop_first=False)

    # Drop any remaining NaN rows
    df = df.dropna()

    # Split features and target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()