# Heart Disease Risk Predictor

A machine learning project that predicts cardiovascular disease risk from patient vitals using 7 classification models — Logistic Regression, SVM, Random Forest (GridSearchCV + Optuna), XGBoost, MLP Neural Network, and Voting Classifier Ensemble.

## Dataset

- Source: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- 918 patients, 11 clinical features
- Target: Heart Disease (0 = No, 1 = Yes)

## Dataset License
The Heart Failure Prediction dataset is publicly available on Kaggle under the Open Database License (ODbL). Free to use for academic and research purposes. Source: fedesoriano (2021). Heart Failure Prediction Dataset. Kaggle.

## Ethical Considerations
- This tool is a screening aid only — not a medical diagnosis
- Model trained on 918 patients — predictions may not generalize to all populations
- Dataset contains more male patients (79%) than female (21%) — model may perform differently across genders (confirmed in slice error analysis: Female 89.5% vs Male 88.4% accuracy)
- No patient identifying information is used — dataset is fully anonymized
- Final medical decisions must always be made by a qualified healthcare professional

## Project Structure

## Models Used

- Logistic Regression (tuned with GridSearchCV)
- Support Vector Machine — RBF kernel (tuned with GridSearchCV)
- Random Forest (tuned with GridSearchCV)
- Random Forest (tuned with Optuna — best model)
- XGBoost (tuned with GridSearchCV)
- MLP Neural Network (tuned with GridSearchCV)
- Voting Classifier (Ensemble of LR + RF Optuna + XGBoost)  

## Results

| Metric               | Logistic Regression | SVM    | Random Forest | XGBoost | MLP    | RF Optuna | Voting Classifier |
| -------------------- | ------------------- | ------ | ------------- | ------- | ------ | --------- | ----------------- |
| Accuracy             | 87%                 | 88%    | 89%           | 87%     | 83%    | 89%       | 89%               |
| ROC-AUC              | 0.903               | 0.898  | 0.930         | 0.925   | 0.888  | 0.927     | —                |
| Heart Disease Recall | 91%                 | 91%    | 91%           | 88%     | 87%    | 92%       | 91%               |
| 10-Fold CV Accuracy  | 83.77%              | 84.18% | 84.46%        | 85.00%  | 83.24% | —        | —                |

## Key Findings

- ST_Slope is the most important predictor of heart disease
- Asymptomatic chest pain (ASY) strongly indicates heart disease
- Optimal classification threshold: 0.405 (improves recall to 94.1%)

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py

# Predict for a new patient 
python predict.py
```

## Tech Stack

Python, scikit-learn, pandas, numpy, matplotlib, seaborn, shap, streamlit

## Live Demo

Run the Streamlit web app locally:

```bash
streamlit run app.py
```
