from src.preprocess import load_and_preprocess
import joblib
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Base project directory (this file's parent folder)
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))


def predict_risk(patient_data, model_name='svm'):
    # Load model and scaler
    if model_name == 'svm':
        model = joblib.load(str(BASE_DIR / 'models' / 'svm_rbf.pkl'))
    else:
        model = joblib.load(
            str(BASE_DIR / 'models' / 'logistic_regression.pkl'))

    scaler = joblib.load(str(BASE_DIR / 'models' / 'scaler.pkl'))

    # Get feature names from preprocessing
    # use the 5000-sample dataset (no SMOTE during feature extraction)
    _, _, _, _, _, feature_names = load_and_preprocess(
        str(BASE_DIR / 'data' / 'data_5000.csv'), apply_smote=False, test_samples=1000)

    # Encode patient data
    df = pd.DataFrame([patient_data])
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    df['ST_Slope'] = df['ST_Slope'].map({'Up': 2, 'Flat': 1, 'Down': 0})
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG'])

    # Align columns with training data
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale and predict
    X_scaled = scaler.transform(df)
    risk_prob = model.predict_proba(X_scaled)[0][1]
    prediction = 'HIGH RISK' if risk_prob >= 0.5 else 'LOW RISK'

    print(f'\n── Heart Disease Risk Assessment ──')
    print(f'Model used:       {model_name.upper()}')
    print(f'Risk probability: {risk_prob:.1%}')
    print(f'Classification:   {prediction}')
    if risk_prob >= 0.7:
        print('⚠ Strongly recommend cardiology referral')
    elif risk_prob >= 0.5:
        print('⚠ Further diagnostic workup advised')
    else:
        print('✓ No immediate intervention indicated')


# Test with a sample patient
if __name__ == '__main__':
    patient = {
        'Age': 57,
        'Sex': 'M',
        'ChestPainType': 'ASY',
        'RestingBP': 130,
        'Cholesterol': 236,
        'FastingBS': 0,
        'RestingECG': 'Normal',
        'MaxHR': 174,
        'ExerciseAngina': 'N',
        'Oldpeak': 0.0,
        'ST_Slope': 'Up'
    }
    predict_risk(patient, model_name='svm')
