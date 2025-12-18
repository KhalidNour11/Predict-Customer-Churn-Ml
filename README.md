Customer Churn Prediction ML Project

Overview

This project predicts customer churn — i.e., customers who are likely to leave a service — using state-of-the-art machine learning models. The goal is to help businesses identify at-risk customers and take preventive actions to retain them.

The project implements multiple models:

XGBoost

LightGBM

CatBoost

Ensemble of all models

It includes exploratory data analysis (EDA), model training, evaluation, hyperparameter tuning, and provides actionable insights for customer retention strategies.

Dataset

This project uses the Telco Customer Churn dataset from Kaggle: Link to dataset
.

The dataset contains 7043 customers with demographic, account, service usage, and billing information. The target variable is:

Churn → Whether the customer left the company (Yes/No)

Key Features

Demographics

gender, SeniorCitizen, Partner, Dependents

Account Information

tenure, Contract, PaymentMethod, PaperlessBilling

Service Usage

PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

Billing

MonthlyCharges, TotalCharges

Target

Churn: Yes/No

Churn prediction is critical for telecom companies, as retaining customers is more cost-effective than acquiring new ones. The dataset includes both numerical and categorical features, making it ideal for real-world ML workflows.

Project Structure
├── images/                # Plots from EDA and model results
├── eda.ipynb              # Exploratory Data Analysis
├── train_xgboost.py       # XGBoost training
├── train_lightgbm.py      # LightGBM training
├── train_catboost.py      # CatBoost training
├── train_ensemble.py      # Ensemble of models
├── helpers.py             # Utility functions
├── params.yaml            # Model hyperparameters
├── requirements.txt       # Python dependencies
├── train.csv              # Training data
├── test.csv               # Test data
└── README.md              # Project documentation

How to Run

Clone the repository

git clone https://github.com/KhalidNour11/Predict-Customer-Churn-Ml.git
cd Predict-Customer-Churn-Ml


Install dependencies

pip install -r requirements.txt


Run EDA

jupyter notebook eda.ipynb


Visualizations include:

Churn distribution

Tenure vs Churn

Monthly Charges vs Churn

Train models

python train_xgboost.py
python train_lightgbm.py
python train_catboost.py
python train_ensemble.py


Evaluate results

Metrics: Accuracy, F1-score, Precision, Recall, ROC-AUC

Confusion matrix and threshold tuning available for each model

Model Performance
Model	Accuracy	F1 Score	ROC-AUC	Notes
XGBoost	0.77	0.77	0.820	Balanced performance, Class 1 lower recall
LightGBM	N/A	N/A	N/A	Model file missing
CatBoost	0.78	0.62	0.830	Best F1 after threshold tuning (0.34)
Ensemble	0.778	0.599	0.831	Slight improvement in ROC-AUC

Confusion Matrix (CatBoost, Threshold=0.34)

[[835 200]
 [116 258]]


ROC Curve Example:


Churn Distribution Plot:


Key Insights

Class 1 (churned customers) is harder to predict → lower F1 and recall.

Threshold tuning improves F1-score for churned customers.

Ensemble slightly improves ROC-AUC, but CatBoost alone gives the best balance.

Imbalanced test set highlights the need for techniques like SMOTE, class weighting, or threshold tuning.

Recommendations

Adjust thresholds per business objective (maximize recall for churned customers).

Perform feature engineering to create interaction or derived features.

Consider ensemble stacking to improve performance.

Document assumptions, preprocessing, and hyperparameters for reproducibility.

Dependencies

Python 3.8+

pandas, numpy

scikit-learn

xgboost, lightgbm, catboost

matplotlib, seaborn

optuna (for hyperparameter tuning)

License

This project is for educational and portfolio purposes.
