import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

DATA_PATH  = os.path.join(BASE_DIR, 'data', 'heart_5000.csv')  # Updated to 5,000 rows
MODELS_DIR = os.path.join(BASE_DIR, 'models')

@st.cache_resource
def load_models():
    # Load all fresh, synchronized pipeline files with memory-mapping to prevent MemoryError
    rf_grid  = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'), mmap_mode='r')
    rf_optuna = joblib.load(os.path.join(MODELS_DIR, 'random_forest_optuna.pkl'), mmap_mode='r')
    
    # Keep the rest standard
    lr       = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    svm      = joblib.load(os.path.join(MODELS_DIR, 'svm_rbf.pkl'))
    xgb      = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))
    mlp      = joblib.load(os.path.join(MODELS_DIR, 'mlp.pkl'))
    voting   = joblib.load(os.path.join(MODELS_DIR, 'voting_classifier.pkl'))
    scaler   = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    
    feature_names = scaler.feature_names_in_.tolist()
    
    return rf_grid, rf_optuna, lr, svm, xgb, mlp, voting, scaler, feature_names

# Update how you unpack them smoothly:
rf_grid, rf_optuna, lr_model, svm_model, xgb_model, mlp_model, voting_model, scaler, feature_names = load_models()

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

# Reordered model choice to spotlight your best ensemble first
model_choice    = st.sidebar.selectbox('Model Selection', [
    'Voting Classifier (Ensemble Framework)',
    'XGBoost Classifier',
    'Random Forest (GridSearchCV)',
    'Random Forest (Optuna Tuned)',
    'Support Vector Machine (SVM)',
    'Logistic Regression',
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
    
    # Mirroring pipeline rules
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

    # 50/50 split: Low Risk (0 to 50%) and High Risk (50% to 100%)
    ax.barh(1, np.pi * 0.50, left=0,          height=0.5, color='#2ecc71', alpha=0.3)
    ax.barh(1, np.pi * 0.50, left=np.pi*0.50, height=0.5, color='#e74c3c', alpha=0.3)

    angle = prob * np.pi
    ax.annotate('', xy=(angle, 0.9), xytext=(angle, 0.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    # Clean two-class labels centered in their respective halves
    ax.text(np.pi * 0.25, 1.4, 'LOW RISK',  ha='center', fontsize=12, color='#2ecc71', fontweight='bold')
    ax.text(np.pi * 0.75, 1.4, 'HIGH RISK', ha='center', fontsize=12, color='#e74c3c', fontweight='bold')
    
    # Display the explicit calculated percentage in the lower center pocket
    ax.text(np.pi * 0.50, 0.3, f'{prob:.1%}', ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig

def plot_model_comparison(X):
    models = {
        'LR': lr_model,
        'SVM': svm_model,
        'RF': rf_optuna,
        'XGBoost': xgb_model,
        'MLP': mlp_model,
        'Voting Ensemble': voting_model
    }
    names = list(models.keys())
    probs = [m.predict_proba(X)[0][1] for m in models.values()]
    colors = ['#e74c3c' if p >= 0.5 else '#2ecc71' for p in probs]

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(names, probs, color=colors, alpha=0.8, edgecolor='white')
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, label='Threshold (0.5)')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Predicted Probability')
    ax.set_title('Cross-Model Prediction Comparison')
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

    if model_choice == 'Voting Classifier (Ensemble Framework)':
        model = voting_model
    elif model_choice == 'XGBoost Classifier':
        model = xgb_model
    elif model_choice == 'Random Forest (GridSearchCV)':
        model = rf_grid
    elif model_choice == 'Random Forest (Optuna Tuned)':
        model = rf_optuna
    elif model_choice == 'Support Vector Machine (SVM)':
        model = svm_model
    elif model_choice == 'Logistic Regression':
        model = lr_model
    else:
        model = mlp_model

    prob = model.predict_proba(X)[0][1]
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
        
        # FIX: Convert the object types to pure strings so PyArrow doesn't crash
        summary['Value'] = summary['Value'].astype(str)
        
        st.dataframe(summary, hide_index=True, width='stretch')

    st.markdown('---')

    # ── Row 2: Model Comparison ──
    st.subheader('Model Comparison')
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
    st.subheader('Explainable AI — Feature Contribution Breakdown (SHAP)')
    
    # We use XGBoost or Random Forest for the explanation to avoid tree-compatibility bugs
    shap_model = xgb_model if model_choice == 'XGBoost Classifier' else rf_optuna
    explainer   = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X)
    
    fig_shap, ax = plt.subplots(figsize=(8, 4))
    
    # Handle single array structures cleanly across XGBoost and sklearn tree outputs
    vals = shap_values[0] if len(shap_values.shape) < 3 else shap_values[0, :, 1]
    
    shap.waterfall_plot(shap.Explanation(
        values        = vals,
        base_values   = explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[1],
        data          = X[0],
        feature_names = feature_names
    ), show=False)
    
    st.pyplot(fig_shap)