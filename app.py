import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

st.markdown("""
<style>
.stApp {
    background-color: #0f1117;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from src.preprocess import load_and_preprocess

DATA_PATH  = os.path.join(BASE_DIR, 'data', 'heart.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

@st.cache_resource
def load_models():
    rf        = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    rf_optuna = joblib.load(os.path.join(MODELS_DIR, 'random_forest_optuna.pkl'))
    lr        = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    svm       = joblib.load(os.path.join(MODELS_DIR, 'svm_rbf.pkl'))
    xgb       = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))
    mlp       = joblib.load(os.path.join(MODELS_DIR, 'mlp.pkl'))
    voting    = joblib.load(os.path.join(MODELS_DIR, 'voting_classifier.pkl'))
    scaler    = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    _, _, _, _, _, feature_names = load_and_preprocess(DATA_PATH)
    return rf, rf_optuna, lr, svm, xgb, mlp, voting, scaler, feature_names

rf_model, rf_optuna_model, lr_model, svm_model, xgb_model, mlp_model, voting_model, scaler, feature_names = load_models()

st.set_page_config(page_title='Heart Disease Risk Predictor', layout='wide')
st.title('Heart Disease Risk Predictor')
st.markdown('Enter patient vitals in the sidebar to predict cardiovascular disease risk.')

st.sidebar.header('Patient Information')
age             = st.sidebar.slider('Age', 20, 80, 50)
sex             = st.sidebar.selectbox('Sex', ['M', 'F'])
chest_pain      = st.sidebar.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
resting_bp      = st.sidebar.slider('Resting Blood Pressure', 80, 200, 120)
cholesterol     = st.sidebar.slider('Cholesterol (mg/dl)', 100, 600, 200)
fasting_bs      = st.sidebar.selectbox('Fasting Blood Sugar > 120', [0, 1])
resting_ecg     = st.sidebar.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
max_hr          = st.sidebar.slider('Max Heart Rate', 60, 202, 150)
exercise_angina = st.sidebar.selectbox('Exercise Angina', ['N', 'Y'])
oldpeak         = st.sidebar.slider('Oldpeak', 0.0, 6.0, 1.0)
st_slope        = st.sidebar.selectbox('ST Slope', ['Up', 'Flat', 'Down'])
model_choice    = st.sidebar.selectbox('Model', [
    'Random Forest Optuna (Best)',
    'Voting Classifier (Ensemble)',
    'Random Forest (GridSearchCV)',
    'XGBoost',
    'Logistic Regression',
    'SVM',
    'MLP Neural Network'
])

def preprocess_input():
    data = {
        'Age': age, 'Sex': sex, 'ChestPainType': chest_pain,
        'RestingBP': resting_bp, 'Cholesterol': cholesterol,
        'FastingBS': fasting_bs, 'RestingECG': resting_ecg,
        'MaxHR': max_hr, 'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak, 'ST_Slope': st_slope
    }
    df = pd.DataFrame([data])
    df['Sex']            = df['Sex'].map({'M': 1, 'F': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    df['ST_Slope']       = df['ST_Slope'].map({'Up': 2, 'Flat': 1, 'Down': 0})
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG'])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return scaler.transform(df)

def plot_gauge(prob):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    # Background arcs
    ax.barh(1, np.pi * 0.33, left=0,          height=0.5, color='#2ecc71', alpha=0.3)
    ax.barh(1, np.pi * 0.33, left=np.pi*0.33, height=0.5, color='#f39c12', alpha=0.3)
    ax.barh(1, np.pi * 0.34, left=np.pi*0.66, height=0.5, color='#e74c3c', alpha=0.3)

    # Needle
    angle = prob * np.pi
    ax.annotate('', xy=(angle, 0.9), xytext=(angle, 0.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    # Labels
    ax.text(np.pi * 0.16, 1.4, 'LOW',    ha='center', fontsize=11, color='#2ecc71', fontweight='bold')
    ax.text(np.pi * 0.50, 1.4, 'MEDIUM', ha='center', fontsize=11, color='#f39c12', fontweight='bold')
    ax.text(np.pi * 0.84, 1.4, 'HIGH',   ha='center', fontsize=11, color='#e74c3c', fontweight='bold')
    ax.text(np.pi * 0.50, 0.3, f'{prob:.1%}', ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig

def plot_model_comparison(X):
    models = {
        'LR': lr_model,
        'SVM': svm_model,
        'RF': rf_model,
        'XGBoost': xgb_model,
        'MLP': mlp_model,
        'RF Optuna': rf_optuna_model,
        'Voting': voting_model
    }
    names = list(models.keys())
    probs = [m.predict_proba(X)[0][1] for m in models.values()]
    colors = ['#e74c3c' if p >= 0.5 else '#2ecc71' for p in probs]

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(names, probs, color=colors, alpha=0.8, edgecolor='white')
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, label='Decision threshold (0.5)')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Predicted probability of heart disease')
    ax.set_title('All 7 models — prediction comparison')
    for bar, prob in zip(bars, probs):
        ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob:.1%}', va='center', fontsize=10)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    return fig

def get_recommendations(prob):
    if prob >= 0.7:
        return {
            'level': '🔴 HIGH RISK',
            'color': 'error',
            'recs': [
                'Immediate cardiology referral strongly recommended',
                'ECG and stress test advised within 1-2 weeks',
                'Review and manage blood pressure and cholesterol levels',
                'Lifestyle changes: diet, exercise, smoking cessation',
                'Do not delay — consult a healthcare professional today'
            ]
        }
    elif prob >= 0.5:
        return {
            'level': '🟡 MODERATE RISK',
            'color': 'warning',
            'recs': [
                'Schedule a cardiology consultation within 1 month',
                'Monitor blood pressure and cholesterol regularly',
                'Consider stress test and full lipid panel blood work',
                'Adopt heart-healthy lifestyle changes',
                'Follow up with primary care physician'
            ]
        }
    else:
        return {
            'level': '🟢 LOW RISK',
            'color': 'success',
            'recs': [
                'Continue routine annual health checkups',
                'Maintain healthy diet and regular exercise',
                'Monitor blood pressure periodically',
                'Avoid smoking and limit alcohol consumption',
                'Re-assess if symptoms develop (chest pain, shortness of breath)'
            ]
        }

if st.button('Predict Risk', type='primary'):
    X = preprocess_input()

    if model_choice == 'Random Forest Optuna (Best)':
        model = rf_optuna_model
    elif model_choice == 'Voting Classifier (Ensemble)':
        model = voting_model
    elif model_choice == 'Random Forest (GridSearchCV)':
        model = rf_model
    elif model_choice == 'XGBoost':
        model = xgb_model
    elif model_choice == 'Logistic Regression':
        model = lr_model
    elif model_choice == 'SVM':
        model = svm_model
    else:
        model = mlp_model

    prob       = model.predict_proba(X)[0][1]
    prediction = 'HIGH RISK' if prob >= 0.5 else 'LOW RISK'

    st.markdown('---')

    # ── Row 1: Gauge + Patient Summary ──
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader('Risk Gauge')
        fig_gauge = plot_gauge(prob)
        st.pyplot(fig_gauge)
        if prob >= 0.7:
            st.error(f'**{prediction}** — {prob:.1%} probability')
        elif prob >= 0.5:
            st.warning(f'**{prediction}** — {prob:.1%} probability')
        else:
            st.success(f'**{prediction}** — {prob:.1%} probability')

    with col2:
        st.subheader('Patient Summary')
        summary = pd.DataFrame({
            'Feature': ['Age', 'Sex', 'Chest Pain Type', 'Resting BP',
                        'Cholesterol', 'Fasting BS', 'Resting ECG',
                        'Max Heart Rate', 'Exercise Angina', 'Oldpeak', 'ST Slope'],
            'Value': [age, sex, chest_pain, resting_bp, cholesterol,
                      fasting_bs, resting_ecg, max_hr, exercise_angina,
                      oldpeak, st_slope]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

    st.markdown('---')

    # ── Row 2: Model Comparison ──
    st.subheader('Model Comparison — All 7 Models')
    fig_compare = plot_model_comparison(X)
    st.pyplot(fig_compare)

    st.markdown('---')

    # ── Row 3: Clinical Recommendations ──
    st.subheader('Clinical Recommendations')
    rec = get_recommendations(prob)
    if rec['color'] == 'error':
        st.error(rec['level'])
    elif rec['color'] == 'warning':
        st.warning(rec['level'])
    else:
        st.success(rec['level'])

    for r in rec['recs']:
        st.markdown(f'- {r}')

    st.caption('⚠️ This tool is a screening aid only — not a medical diagnosis. Always consult a qualified healthcare professional.')

    st.markdown('---')

    # ── Row 4: SHAP Explanation ──
    st.subheader('Why did the model predict this?')
    explainer   = shap.TreeExplainer(rf_optuna_model)
    shap_values = explainer.shap_values(X)
    fig_shap, ax = plt.subplots()
    shap.waterfall_plot(shap.Explanation(
        values        = shap_values[0, :, 1],
        base_values   = explainer.expected_value[1],
        data          = X[0],
        feature_names = feature_names
    ), show=False)
    st.pyplot(fig_shap)