import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'F:\AI\HDRP\HDRP')
from src.preprocess import load_and_preprocess

# Load model and data
rf_model = joblib.load(r'F:\AI\HDRP\HDRP\models\random_forest.pkl')
X_train, X_test, y_train, y_test, scaler, feature_names = \
    load_and_preprocess(r'F:\AI\HDRP\HDRP\data\heart.csv')

# Create SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Plot 1 — Summary plot (all features ranked by importance)
plt.figure()
shap.summary_plot(shap_values[:, :, 1], X_test, 
                  feature_names=feature_names, show=False)
plt.title('SHAP Summary Plot — Random Forest')
plt.tight_layout()
plt.savefig(r'F:\AI\HDRP\HDRP\models\shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 2 — Single patient explanation (first test patient)
plt.figure()
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0, :, 1],
    base_values=explainer.expected_value[1],
    data=X_test[0],
    feature_names=feature_names
))
plt.tight_layout()
plt.savefig(r'F:\AI\HDRP\HDRP\models\shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.show()

print('SHAP plots saved to models/')