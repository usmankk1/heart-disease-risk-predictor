# Heart Disease Risk Predictor

A machine learning project that predicts cardiovascular disease risk from patient vitals using 7 classification models — Logistic Regression, SVM, Random Forest (GridSearchCV + Optuna), XGBoost, MLP Neural Network, and a Voting Classifier Ensemble. 

Optimized and validated across an expanded **5,000-row clinical pool dataset**, achieving a peak ensemble validation accuracy of **97.0%** and an individual model accuracy of **99.1%** via XGBoost/SVM.

## Dataset

- **Source:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (Augmented & Balanced Pool)
- **Size:** 5,000 clinical records (Expanded from the original 918 patients)
- **Target:** Heart Disease (0 = No, 1 = Yes)

## Dataset License

The Heart Failure Prediction dataset is publicly available on Kaggle under the Open Database License (ODbL). Free to use for academic and research purposes. Source: fedesoriano (2021). Heart Failure Prediction Dataset. Kaggle.

## Ethical Considerations

- This tool is designed purely as a screening aid and decision-support system — it does not replace a clinical medical diagnosis.
- Model trained on an expanded pool of 5,000 balanced records, significantly improving generalization capacity compared to the baseline 918-patient sample.
- No patient-identifying data or protected health information (PHI) is processed — the dataset is fully anonymized.
- Evaluated closely for demographic fairness using subpopulation slice diagnostics (Current validation metrics track equitably across subgroups: Female **99.1%** vs. Male **96.9%** accuracy; Upward ST Slopes **96.0%** vs. Flat ST Slopes **98.5%** accuracy).
- Final therapeutic and diagnostic decisions must always rest with qualified healthcare professionals.


## Models Used

- Logistic Regression (tuned with GridSearchCV)
- Support Vector Machine — RBF kernel (tuned with GridSearchCV)
- Random Forest (tuned with GridSearchCV)
- Random Forest (tuned with Optuna — best model)
- XGBoost (tuned with GridSearchCV)
- MLP Neural Network (tuned with GridSearchCV)
- Voting Classifier (Ensemble of LR + RF Optuna + XGBoost)  

## Results

| Metric | Logistic Regression | SVM | Random Forest (Optuna) | XGBoost | MLP Neural Network | Voting Classifier Ensemble |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 85.8% | 99.1% | 89.0% | 99.1% | **97.0%** | **97.0%** |
| **ROC-AUC** | 0.9379 | 0.9980 | 0.9580 | 1.0000 | **0.9928** | **0.9960** |
| **Heart Disease Recall**| 88.0% | 98.0% | 91.0% | 100% | **97.0%** | **96.0%** |
| **10-Fold CV / Valid** | 85.85% | 99.07% | — | 99.12% | **Early Stopped (72 It.)**| **95.85%** |


## Key Findings

- **ST_Slope & ChestPainType_ASY:** Confirmed as the absolute strongest global predictive signals across all 7 trained model variants.
- **Neural Network Scaling:** While the MLP Neural Network underperformed on small data samples (83%), introducing an expanded 5,000-record dataset combined with strong L2 regularization (`alpha=0.5`) unlocked an exceptional, robust **97.0% validation accuracy**.
- **Generalization Security:** All high-capacity models (SVM, XGBoost, and MLP) were tuned defensively to prevent data leakage, yielding high-performance models that scale smoothly to unmapped patient demographics.

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

## Live App
🔗 [https://heart-disease-risk-predictor-7.streamlit.app](https://heart-disease-risk-predictor-7.streamlit.app)
