"""
app.py
──────
Heart Disease Risk Predictor - Streamlit Web Application

Author: Usman Khan, Ibrahim, Abdul Hannan
SAP IDs: 37906, 24865, 39115
Course: AI
"""
from matplotlib.projections.polar import PolarAxes
from typing import cast
from src.preprocess import load_and_preprocess
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import sys
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

st.set_page_config(page_title='Heart Disease Risk Predictor', layout='wide')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Use the 5000-sample dataset by default
DATA_PATH = os.path.join(BASE_DIR, 'data', 'data_5000.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def safe_predict_proba(model, X):
    if hasattr(model, 'predict_proba'):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass

    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X)
        if scores.ndim > 1 and scores.shape[1] > 1:
            scores = scores[:, 1]
        min_score, max_score = scores.min(), scores.max()
        if np.isclose(max_score, min_score):
            return np.zeros_like(scores, dtype=float)
        return (scores - min_score) / (max_score - min_score)

    return np.zeros(X.shape[0], dtype=float)


def build_shap_explainer(model, X_background):
    if hasattr(model, 'predict_proba'):
        def f(x):
            return model.predict_proba(x)[:, 1]
    elif hasattr(model, 'decision_function'):
        def f(x):
            scores = model.decision_function(x)
            if scores.ndim > 1 and scores.shape[1] > 1:
                scores = scores[:, 1]
            min_score, max_score = scores.min(), scores.max()
            return (scores - min_score) / (max_score - min_score)
    else:
        raise ValueError('Model does not support probability or decision scores for SHAP interpretation.')

    return shap.Explainer(f, X_background, feature_names=feature_names)


@st.cache_resource
def load_models():
    rf = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    rf_optuna = joblib.load(os.path.join(
        MODELS_DIR, 'random_forest_optuna.pkl'))
    lr = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    svm = joblib.load(os.path.join(MODELS_DIR, 'svm_rbf.pkl'))
    xgb = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))
    mlp = joblib.load(os.path.join(MODELS_DIR, 'mlp.pkl'))
    voting = joblib.load(os.path.join(MODELS_DIR, 'voting_classifier.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

    X_train, X_test, y_train, y_test, _, feature_names = load_and_preprocess(
        DATA_PATH, apply_smote=False, test_samples=1000)

    models = {
        'Logistic Regression': lr,
        'SVM': svm,
        'Random Forest': rf,
        'XGBoost': xgb,
        'MLP Neural Network': mlp,
        'Random Forest Optuna': rf_optuna,
        'Voting Classifier': voting
    }

    performance_records = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = safe_predict_proba(model, X_test)
        performance_records.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_proba) if np.unique(y_proba).size > 1 else float('nan')
        })

    performance_df = pd.DataFrame(performance_records)
    return rf, rf_optuna, lr, svm, xgb, mlp, voting, scaler, feature_names, X_train, X_test, y_train, y_test, performance_df


rf_model, rf_optuna_model, lr_model, svm_model, xgb_model, mlp_model, voting_model, scaler, feature_names, X_train, X_test, y_train, y_test, performance_df = load_models()

MODEL_LOOKUP = {
    'Random Forest Optuna (Best)': rf_optuna_model,
    'Voting Classifier (Ensemble)': voting_model,
    'Random Forest (GridSearchCV)': rf_model,
    'XGBoost': xgb_model,
    'Logistic Regression': lr_model,
    'SVM': svm_model,
    'MLP Neural Network': mlp_model
}


st.title('Heart Disease Risk Predictor')
st.markdown(
    'Enter patient vitals in the sidebar to predict cardiovascular disease risk.')

st.sidebar.header('Patient Information')
age = st.sidebar.slider('Age', 20, 80, 50)
sex = st.sidebar.selectbox('Sex', ['M', 'F'])
chest_pain = st.sidebar.selectbox(
    'Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.sidebar.slider('Resting Blood Pressure', 80, 200, 120)
cholesterol = st.sidebar.slider('Cholesterol (mg/dl)', 100, 600, 200)
fasting_bs = st.sidebar.selectbox('Fasting Blood Sugar > 120', [0, 1])
resting_ecg = st.sidebar.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
max_hr = st.sidebar.slider('Max Heart Rate', 60, 202, 150)
exercise_angina = st.sidebar.selectbox('Exercise Angina', ['N', 'Y'])
oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.0, 1.0)
st_slope = st.sidebar.selectbox('ST Slope', ['Up', 'Flat', 'Down'])
model_choice = st.sidebar.selectbox('Model', [
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
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    df['ST_Slope'] = df['ST_Slope'].map({'Up': 2, 'Flat': 1, 'Down': 0})
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG'])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return scaler.transform(df)


def plot_gauge(prob):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})
    ax = cast(PolarAxes, ax)
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    # Background arcs - LOW and HIGH only
    ax.barh(1, np.pi * 0.5, left=0,
            height=0.5, color='#2ecc71', alpha=0.3)
    ax.barh(1, np.pi * 0.5, left=np.pi*0.5,
            height=0.5, color='#e74c3c', alpha=0.3)

    # Needle
    angle = prob * np.pi
    ax.annotate('', xy=(angle, 0.9), xytext=(angle, 0.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    ax.text(np.pi * 0.25, 1.4, 'LOW',  ha='center',
            fontsize=13, color='#2ecc71', fontweight='bold')
    ax.text(np.pi * 0.75, 1.4, 'HIGH', ha='center',
            fontsize=13, color='#e74c3c', fontweight='bold')
    ax.text(np.pi * 0.50, 0.3, f'{prob:.1%}',
            ha='center', fontsize=14, fontweight='bold')

    ax.set_title('Risk Gauge')
    plt.tight_layout()
    return fig


def plot_model_comparison(X):
    models = {
        'Logistic Regression': lr_model,
        'SVM': svm_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'MLP Neural Network': mlp_model,
        'Random Forest Optuna': rf_optuna_model,
        'Voting Classifier': voting_model
    }
    names = list(models.keys())
    probs = [safe_predict_proba(m, X)[0] for m in models.values()]
    colors = ['#e74c3c' if p >= 0.5 else '#2ecc71' for p in probs]

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(names, probs, color=colors, alpha=0.8, edgecolor='white')
    ax.axvline(x=0.5, color='gray', linestyle='--',
               linewidth=1, label='Decision threshold (0.5)')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Predicted probability of heart disease')
    ax.set_title('All 7 Models - Prediction Comparison')
    for bar, prob in zip(bars, probs):
        ax.text(min(prob + 0.02, 1.0), bar.get_y() + bar.get_height()/2,
                f'{prob:.1%}', va='center', fontsize=10)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    return fig


def get_recommendations(prob):
    if prob >= 0.5:
        return {
            'level': 'HIGH RISK',
            'color': 'error',
            'recs': [
                'Immediate cardiology referral strongly recommended',
                'ECG and stress test advised within 1-2 weeks',
                'Review and manage blood pressure and cholesterol levels',
                'Lifestyle changes: diet, exercise, smoking cessation',
                'Do not delay - consult a healthcare professional today'
            ]
        }
    else:
        return {
            'level': 'LOW RISK',
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
    model = MODEL_LOOKUP[model_choice]
    prob = safe_predict_proba(model, X)[0]
    prediction = 'HIGH RISK' if prob >= 0.5 else 'LOW RISK'

    st.markdown('---')

    # -- Row 1: Gauge + Patient Summary --
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader('Risk Gauge')
        fig_gauge = plot_gauge(prob)
        st.pyplot(fig_gauge)
        plt.close()
        if prob >= 0.5:
            st.error(f'**{prediction}** - {prob:.1%} probability')
        else:
            st.success(f'**{prediction}** - {prob:.1%} probability')

        st.caption('Prediction is based on model probability and a 0.5 threshold.')

    with col2:
        st.subheader('Patient Summary')
        summary = pd.DataFrame({
            'Feature': ['Age', 'Sex', 'Chest Pain Type', 'Resting BP',
                        'Cholesterol', 'Fasting BS', 'Resting ECG',
                        'Max Heart Rate', 'Exercise Angina', 'Oldpeak', 'ST Slope'],
            'Value': [str(age), str(sex), str(chest_pain), str(resting_bp),
                      str(cholesterol), str(fasting_bs), str(resting_ecg),
                      str(max_hr), str(exercise_angina), str(oldpeak), str(st_slope)]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

    st.markdown('---')

    # -- Row 2: Model Comparison --
    st.subheader('Model Comparison - All 7 Models')
    fig_compare = plot_model_comparison(X)
    st.pyplot(fig_compare)
    plt.close()

    st.markdown('---')

    # -- Row 3: Clinical Recommendations --
    st.subheader('Clinical Recommendations')
    rec = get_recommendations(prob)
    if rec['color'] == 'error':
        st.error(f"🔴 {rec['level']}")
    else:
        st.success(f"🟢 {rec['level']}")

    for r in rec['recs']:
        st.markdown(f'- {r}')

    st.caption('This tool is a screening aid only - not a medical diagnosis. Always consult a qualified healthcare professional.')

    st.markdown('---')

    # -- Row 4: SHAP Explanation --
    st.subheader('Why did the model predict this?')
    try:
        explainer = build_shap_explainer(model, X_train)
        shap_values = explainer(X)

        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(plt.gcf())
        plt.close()
    except Exception as e:
        st.warning('SHAP explanation is unavailable for this model: ' + str(e))
        st.write('Selected model may not support a consistent SHAP explainer with the current backend.')

    st.markdown('---')

    # -- Row 5: Model Performance --
    st.subheader('Model Performance on Held-Out Test Set')
    st.dataframe(performance_df.set_index('Model').style.format({
        'Accuracy': '{:.3f}',
        'Precision': '{:.3f}',
        'Recall': '{:.3f}',
        'F1 Score': '{:.3f}',
        'ROC-AUC': '{:.3f}'
    }), use_container_width=True,)
