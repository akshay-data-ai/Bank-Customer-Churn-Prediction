# 🏦 Bank Customer Churn Prediction Using Machine Learning

## Project Overview

An end-to-end Machine Learning project designed to analyze bank customer behavior and predict customer churn.

The project includes Exploratory Data Analysis (EDA), preprocessing, model training, model evaluation, cross-validation, threshold tuning, feature importance analysis, and an interactive Streamlit application.


## 🚀 Live Streamlit Application

Try the deployed Bank Customer Churn Prediction app:

🔗 Live Application : https://bank-customer-churn-ai.streamlit.app/

Enter customer details to generate a churn risk score
using our trained Gradient Boosting model.

**Selected Model:** Gradient Boosting

**Classification Threshold:** 0.30

**Recall:** 65.36%

**F1-Score:** 64.41%

*Metrics are from exploratory test evaluation.*

## Business Objective

Identify customers who are at risk of leaving the bank and provide data-driven insights to support customer retention strategies.

## Dataset

- **Source:** [Bank Customer Churn Dataset — Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
- **Total Records:** 10,000
- **Total Columns:** 12
- **Target Variable:** `churn`
- **0:** Customer Stayed
- **1:** Customer Left

## Technologies Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Joblib, Google Colab, Streamlit and GitHub.

## Exploratory Data Analysis

Key observations:

- Overall customer churn rate: 20.37%
- Inactive customers show higher churn than active customers.
- Germany has the highest observed churn rate among the countries analyzed.
- Customers holding two banking products have comparatively low churn.
- Age, product count, membership activity and account balance are important model features.

## Machine Learning Models

Five classification algorithms were evaluated.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.80% | 58.91% | 18.67% | 28.36% |
| Random Forest | 86.05% | 77.83% | 43.98% | 56.20% |
| Gradient Boosting | 87.00% | 79.28% | 48.89% | 60.49% |
| XGBoost | 86.70% | 78.78% | 47.42% | 59.20% |
| Extra Trees | 85.75% | 76.52% | 43.24% | 55.26% |

## Selected Model — Gradient Boosting

Gradient Boosting was selected as the preferred candidate based on cross-validation performance.

Classification threshold tuning was performed to improve Recall and F1-Score.

**Tuned model results (threshold = 0.30):**

- Accuracy: 85.30%
- Precision: 63.48%
- Recall: 65.36%
- F1-Score: 64.41%

These are exploratory test-set results. The test set was previously used during model comparisons, so an independent final evaluation is still required.

## Streamlit Application

The Streamlit application allows users to:

- Enter customer banking information.
- Generate a model-based churn score.
- View a prediction indicating whether a customer is likely to stay or leave.
- Explore customer churn using the saved Gradient Boosting model.

## How to Run

1. Clone or download this repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Launch the application using `streamlit run app.py`.
4. Open the local Streamlit address displayed in the terminal.

To reproduce the machine learning analysis, open `Bank_Customer_Churn_Prediction.ipynb` in Google Colab and follow its dataset-loading instructions.

## Project Files

- `Bank_Customer_Churn_Prediction.ipynb` — Complete machine learning notebook
- `Bank Customer Churn Prediction.csv` — Dataset
- `app.py` — Streamlit application
- `bank_customer_churn_model.pkl` — Saved trained model
- `requirements.txt` — Python dependencies

## Limitations

This project is for educational and portfolio demonstration purposes. Model probabilities are not yet independently calibrated or validated for production use. Real-world banking applications would require additional testing, data privacy safeguards and fairness evaluation.

## Future Enhancements

- Additional model tuning and probability calibration
- Business impact analysis of retention campaigns
- Improved Streamlit dashboard visualizations
- Independent model validation

## Author

**akshay-data-ai**

Machine Learning | Python | Banking Analytics
