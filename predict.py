import joblib
import pandas as pd
import numpy as np
import sys
sys.path.append(r'F:\AI\HDRP\HDRP')

# Set configuration paths centrally
DATA_PATH = r'F:\AI\HDRP\HDRP\data\heart.csv'  # Reverted to original 918-patient real dataset
MODEL_DIR = r'F:\AI\HDRP\HDRP\models'

def predict_risk(patient_data, model_name='svm'):
    # 1. Load model and scaler
    if model_name.lower() == 'svm':
        model = joblib.load(rf'{MODEL_DIR}\svm_rbf.pkl')
    elif model_name.lower() == 'voting':
        model = joblib.load(rf'{MODEL_DIR}\voting_classifier.pkl')
    else:
        model = joblib.load(rf'{MODEL_DIR}\logistic_regression.pkl')
    
    scaler = joblib.load(rf'{MODEL_DIR}\scaler.pkl')

    # 2. Extract feature names directly from the fitted scaler (Instant!)
    feature_names = scaler.feature_names_in_.tolist()

    # 3. Structuring and Cleaning Input Data
    df = pd.DataFrame([patient_data])
    
    # Mirror the training pipeline's zero-value handling
    if df['Cholesterol'].iloc[0] == 0 or pd.isna(df['Cholesterol'].iloc[0]):
        raw_df = pd.read_csv(DATA_PATH)
        df['Cholesterol'] = raw_df['Cholesterol'].replace(0, np.nan).median()
        
    if df['RestingBP'].iloc[0] == 0 or pd.isna(df['RestingBP'].iloc[0]):
        raw_df = pd.read_csv(DATA_PATH)
        df['RestingBP'] = raw_df['RestingBP'].replace(0, np.nan).median()

    # 4. Encode patient data
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    df['ST_Slope'] = df['ST_Slope'].map({'Up': 2, 'Flat': 1, 'Down': 0})
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG'])

    # 5. Align columns with training dataset rules
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # 6. Scale and predict
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
    predict_risk(patient, model_name='voting')