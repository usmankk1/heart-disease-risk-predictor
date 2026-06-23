# Heart Disease Risk Predictor

A machine learning project that predicts cardiovascular disease risk from patient vitals using 7 classification models — Logistic Regression, SVM, Random Forest (GridSearchCV + Optuna), XGBoost, MLP Neural Network, and Voting Classifier Ensemble.

## Dataset

- Source: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- 918 real patients, 11 clinical features — augmented to 5000 samples using SMOTE
- Training set: 4000 samples (SMOTE augmented, balanced 50/50)
- Test set: 1000 real clinical samples (no synthetic data — no leakage)
- Target: Heart Disease (0 = No, 1 = Yes)

## Dataset License
The Heart Failure Prediction dataset is publicly available on Kaggle under the Open Database License (ODbL). Free to use for academic and research purposes. Source: fedesoriano (2021). Heart Failure Prediction Dataset. Kaggle.

## Ethical Considerations
- This tool is a screening aid only — not a medical diagnosis
- Base dataset contains 918 real patients — SMOTE augmentation used to reach 5000 samples for improved generalization
- Dataset contains more male patients (79%) than female (21%) — model may perform differently across genders
- No patient identifying information is used — dataset is fully anonymized
- Final medical decisions must always be made by a qualified healthcare professional

## Project Structure

## Models Used

- Logistic Regression (tuned with GridSearchCV)
- Support Vector Machine — RBF kernel (tuned with GridSearchCV)
- Random Forest (tuned with GridSearchCV)
- Random Forest (tuned with Optuna — Bayesian optimisation)
- XGBoost (tuned with GridSearchCV)
- MLP Neural Network (tuned with GridSearchCV)
- Voting Classifier (Ensemble of LR + RF Optuna + XGBoost)

## Results

| Metric               | Logistic Regression | SVM    | Random Forest | XGBoost | MLP    | RF Optuna | Voting Classifier |
| -------------------- | ------------------- | ------ | ------------- | ------- | ------ | --------- | ----------------- |
| Accuracy             | 87.6%               | 95.3%  | 96.4%         | 97.5%   | 97.3%  | 95.4%     | 96.4%             |
| ROC-AUC              | 0.9474              | 0.9874 | 0.9946        | 0.9958  | 0.9932 | 0.9932    | 0.9925            |
| Heart Disease Recall | 88%                 | 94%    | 95%           | 98%     | 97%    | 94%       | 96%               |
| 10-Fold CV F1        | 88.92%              | 94.47% | 96.43%        | 97.30%  | 96.40% | 95.95%    | 96.48%            |

## Key Finding
- XGBoost is the best performing model with 97.5% accuracy and 0.9958 ROC-AUC
- ST_Slope is the most important predictor of heart disease
- Asymptomatic chest pain (ASY) strongly indicates heart disease
- SMOTE augmentation significantly improved model performance across all models

## How to Run

### Prerequisites
- Python 3.9+
- pip (Python package manager)

### Setup

1. **Clone or navigate to the project:**
   ```bash
   cd hdrp
   ```

2. **Create and activate virtual environment (recommended):**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

**Full ML Pipeline (Train & Evaluate all 7 models):**
```bash
python main.py
```

**Quick Baseline Training:**
```bash
python src/quick_train.py
```

**Generate SHAP Explanations:**
```bash
python src/shap_explain.py
```

**Run Streamlit Web App:**
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

**Make Individual Predictions:**
```bash
python predict.py
```

**Run Tests:**
```bash
pytest tests/ -v --cov=src
```

### Project Scripts

| Script | Purpose |
|--------|---------|
| `main.py` | Full pipeline: preprocess → train all 7 models → evaluate → save |
| `app.py` | Streamlit web UI for risk prediction & SHAP explanations |
| `predict.py` | Standalone prediction for new patients |
| `src/train.py` | Model training with hyperparameter tuning |
| `src/evaluate.py` | Model evaluation metrics & ROC curves |
| `src/preprocess.py` | Data loading, cleaning, SMOTE, scaling |
| `src/balanced_train.py` | Train subset of models with balanced hyperparameters |
| `src/run_svm.py` | Train SVM model only |
| `src/save_report.py` | Generate markdown/text evaluation reports |
| `src/shap_explain.py` | Generate SHAP waterfall plots for interpretability |

## Tech Stack

Python, scikit-learn, xgboost, optuna, imbalanced-learn, pandas, numpy, matplotlib, seaborn, shap, streamlit

## Live Demo

Run the Streamlit web app locally:

```bash
streamlit run app.py
```

## Live App
🔗 [https://heart-disease-risk-predictor-7.streamlit.app](https://heart-disease-risk-predictor-7.streamlit.app)