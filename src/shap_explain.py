import shap
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Perfect choice for headless plot generation
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(r'F:\AI\HDRP\HDRP')
from src.preprocess import load_and_preprocess

# Centralized Path Variables
DATA_PATH = r'F:\AI\HDRP\HDRP\data\heart.csv'  # Reverted to original 918-patient real dataset
MODEL_DIR = r'F:\AI\HDRP\HDRP\models'

# 1. Load the Optuna-tuned Random Forest (our best model) and preprocess matching data split
rf_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_optuna.pkl'))
X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(DATA_PATH)

# 2. Initialize the TreeExplainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Handle potential structural dimension shapes across scikit-learn tree outputs smoothly
if len(shap_values.shape) == 3:
    # If shape is (samples, features, classes), grab class 1 (Heart Disease)
    shap_vals_summary = shap_values[:, :, 1]
    shap_vals_waterfall = shap_values[0, :, 1]
    base_val = explainer.expected_value[1]
else:
    # Fallback if binary output format is compressed
    shap_vals_summary = shap_values
    shap_vals_waterfall = shap_values[0]
    base_val = explainer.expected_value

# ── Plot 1: Summary ──────────────────────────────────────────
plt.figure()
shap.summary_plot(shap_vals_summary, X_test, feature_names=feature_names, show=False)
plt.title('SHAP Summary Plot — Random Forest', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Summary plot saved successfully.')

# ── Plot 2: Waterfall ────────────────────────────────────────
plt.figure()
shap.waterfall_plot(shap.Explanation(
    values=shap_vals_waterfall,
    base_values=base_val,
    data=X_test[0],
    feature_names=feature_names
), show=False)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'shap_waterfall.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Waterfall plot saved successfully.')

print(f'\n✓ Both SHAP plots safely exported to: {MODEL_DIR}')