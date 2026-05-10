import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import sys

sys.path.append(r'F:\AI\HDRP\HDRP')
from src.preprocess import load_and_preprocess

@st.cache_resource
def load_models():
    rf = joblib.load(r'F:\AI\HDRP\HDRP\models\random_forest.pkl')
    lr = joblib.load(r'F:\AI\HDRP\HDRP\models\logistic_regression.pkl')
    svm = joblib.load(r'F:\AI\HDRP\HDRP\models\svm_rbf.pkl')
    scaler = joblib.load(r'F:\AI\HDRP\HDRP\models\scaler.pkl')
    _, _, _, _, _, feature_names = load_and_preprocess(r'F:\AI\HDRP\HDRP\data\heart.csv')
    return rf, lr, svm, scaler, feature_names

rf_model, lr_model, svm_model, scaler, feature_names = load_models()

st.title('Heart Disease Risk Predictor')
st.markdown('Enter patient vitals below to predict cardiovascular disease risk.')

st.sidebar.header('Patient Information')
age = st.sidebar.slider('Age', 20, 80, 50)
sex = st.sidebar.selectbox('Sex', ['M', 'F'])
chest_pain = st.sidebar.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.sidebar.slider('Resting Blood Pressure', 80, 200, 120)
cholesterol = st.sidebar.slider('Cholesterol (mg/dl)', 100, 600, 200)
fasting_bs = st.sidebar.selectbox('Fasting Blood Sugar > 120', [0, 1])
resting_ecg = st.sidebar.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
max_hr = st.sidebar.slider('Max Heart Rate', 60, 202, 150)
exercise_angina = st.sidebar.selectbox('Exercise Angina', ['N', 'Y'])
oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.0, 1.0)
st_slope = st.sidebar.selectbox('ST Slope', ['Up', 'Flat', 'Down'])
model_choice = st.sidebar.selectbox('Model', ['Random Forest', 'Logistic Regression', 'SVM'])

def preprocess_input():
    data = {
        'Age': age, 'Sex': sex, 'ChestPainType': chest_pain,
        'RestingBP': resting_bp, 'Cholesterol': cholesterol,
        'FastingBS': fasting_bs, 'RestingECG': resting_ecg,
        'MaxHR': max_hr, 'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak, 'ST_Slope': st_slope
    }
    df = pd.DataFrame([data])
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    df['ST_Slope'] = df['ST_Slope'].map({'Up': 2, 'Flat': 1, 'Down': 0})
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG'])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return scaler.transform(df)

if st.button('Predict Risk'):
    X = preprocess_input()
    if model_choice == 'Random Forest':
        model = rf_model
    elif model_choice == 'Logistic Regression':
        model = lr_model
    else:
        model = svm_model
    prob = model.predict_proba(X)[0][1]
    prediction = 'HIGH RISK' if prob >= 0.5 else 'LOW RISK'
    st.markdown('---')
    if prob >= 0.5:
        st.error(f'**{prediction}** — {prob:.1%} probability of heart disease')
    else:
        st.success(f'**{prediction}** — {prob:.1%} probability of heart disease')
    st.subheader('Why did the model predict this?')
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0, :, 1],
        base_values=explainer.expected_value[1],
        data=X[0],
        feature_names=feature_names
    ), show=False)
    st.pyplot(fig)