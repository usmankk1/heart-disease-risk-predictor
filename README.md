# Heart Disease Risk Predictor
A machine learning project that predicts cardiovascular disease risk from patient vitals using Classification algorithms (Logistic Regression, SVM, and Random Forest).

## Dataset
- Source: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- 918 patients, 11 clinical features
- Target: Heart Disease (0 = No, 1 = Yes)

## Project Structure

## Models Used
## Models Used
- Logistic Regression (tuned with GridSearchCV)
- Support Vector Machine — RBF kernel (tuned with GridSearchCV)
- Random Forest (tuned with GridSearchCV)
- Random Forest (tuned with Optuna — best model)
- XGBoost (tuned with GridSearchCV)
- MLP Neural Network (tuned with GridSearchCV)
- Voting Classifier (Ensemble of LR + RF Optuna + XGBoost)

## Results
| Metric | Logistic Regression | SVM | Random Forest |
|--------|-------------------|-----|---------------|
| Accuracy | 87% | 88% | 89% |
| ROC-AUC | 0.903 | 0.898 | 0.930 |
| Heart Disease Recall | 91% | 91% | 91% |
| 10-Fold CV Accuracy | 83.77% | 84.18% | 84.46% |

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
Enter patient vitals in the sidebar, click **Predict Risk** to get an instant risk assessment with SHAP explanation.