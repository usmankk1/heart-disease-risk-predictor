from src.preprocess import load_and_preprocess
import sys
import matplotlib.pyplot as plt
import shap
import joblib
import numpy as np
import matplotlib
from pathlib import Path
matplotlib.use('Agg')

# Use project-relative paths
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / 'models'
DATA_PATH = ROOT / 'data' / 'data_5000.csv'


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
        return shap.Explainer(model.predict_proba, X_background, feature_names=feature_names)

    if hasattr(model, 'decision_function'):
        def f(x):
            scores = model.decision_function(x)
            if scores.ndim > 1 and scores.shape[1] > 1:
                scores = scores[:, 1]
            min_score, max_score = scores.min(), scores.max()
            if np.isclose(max_score, min_score):
                return np.zeros_like(scores, dtype=float)
            return (scores - min_score) / (max_score - min_score)

        return shap.Explainer(f, X_background, feature_names=feature_names)

    raise ValueError('Model does not support SHAP explanation via predict_proba or decision_function.')


def save_shap_plots(model, model_name, X_background, X_explain, feature_names):
    explainer = build_shap_explainer(model, X_background)
    shap_values = explainer(X_explain)

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary Plot — {model_name}')
    plt.tight_layout()
    summary_path = MODELS_DIR / f'shap_summary_{model_name.replace(" ", "_").lower()}.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Summary plot saved to {summary_path}')

    # Waterfall plot for the first observation
    plt.figure(figsize=(10, 6))
    try:
        shap.plots.waterfall(shap_values[0], show=False)
    except Exception:
        shap.waterfall_plot(shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=X_explain[0],
            feature_names=feature_names
        ), show=False)

    plt.tight_layout()
    waterfall_path = MODELS_DIR / f'shap_waterfall_{model_name.replace(" ", "_").lower()}.png'
    plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Waterfall plot saved to {waterfall_path}')


if __name__ == '__main__':
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the dataset and model
    rf_model = joblib.load(MODELS_DIR / 'random_forest.pkl')
    rf_optuna_model = joblib.load(MODELS_DIR / 'random_forest_optuna.pkl')
    X_train, X_test, y_train, y_test, scaler, feature_names = \
        load_and_preprocess(str(DATA_PATH), apply_smote=False, test_samples=1000)

    for model, name in [(rf_model, 'Random Forest'), (rf_optuna_model, 'Random Forest Optuna')]:
        try:
            save_shap_plots(model, name, X_train, X_test[:50], feature_names)
        except Exception as exc:
            print(f'Could not generate SHAP plots for {name}: {exc}')

    print('SHAP explanation artifacts generation complete.')
