Customer Churn Prediction ML Project

Overview

This project predicts customer churn (customers likely to leave a service) using multiple machine learning models. The aim is to help businesses identify at-risk customers and retain them.

Models used:

XGBoost

LightGBM

CatBoost

Ensemble

The project includes EDA, model training, evaluation, and hyperparameter tuning.

Dataset Description

Telco Customer Churn dataset (Kaggle link
) contains customer demographics, account information, service usage, and billing data.

Target variable: Churn (Yes = customer left, No = customer stayed)

Key Features:

Demographics: gender, SeniorCitizen, Partner, Dependents

Account Info: tenure, Contract, PaymentMethod, PaperlessBilling

Service Usage: PhoneService, MultipleLines, InternetService, OnlineSecurity, StreamingTV, etc.

Billing: MonthlyCharges, TotalCharges

Importance:
Churn prediction allows companies to take preventive measures to reduce customer loss. Dataset contains both numerical and categorical features, suitable for ML pipelines.

Project Structure
├── images/                # Plots and graphs from EDA and results
├── eda.ipynb              # Exploratory Data Analysis
├── train_xgboost.py       # XGBoost model training
├── train_lightgbm.py      # LightGBM model training
├── train_catboost.py      # CatBoost model training
├── train_ensemble.py      # Ensemble of all models
├── helpers.py             # Helper functions
├── params.yaml            # Hyperparameters
├── requirements.txt       # Required Python packages
├── train.csv              # Training data
├── test.csv               # Test data
├── customer_churn_ml.zip  # Data & project archive
└── README.md              # Project documentation

How to Run

Clone the repository

git clone https://github.com/KhalidNour11/Predict-Customer-Churn-Ml.git
cd Predict-Customer-Churn-Ml


Install dependencies

pip install -r requirements.txt


Run EDA

jupyter notebook eda.ipynb


Plots include: churn distribution, tenure vs churn, monthly charges vs churn.

Train models

python train_xgboost.py
python train_lightgbm.py
python train_catboost.py
python train_ensemble.py


Evaluate results

Metrics: Accuracy, F1-score, Precision, Recall, ROC-AUC

Confusion matrices and threshold tuning are included

Model Performance
Model	Accuracy	F1 Score	ROC-AUC	Notes
XGBoost	0.77	0.77	0.820	Good overall, Class 1 lower recall
LightGBM	N/A	N/A	N/A	Model file missing in test
CatBoost	0.78	0.62	0.830	Best F1 after threshold tuning (0.34)
Ensemble	0.778	0.599	0.831	Slight ROC-AUC improvement

Confusion Matrix (CatBoost, Threshold=0.34)

[[835 200]
 [116 258]]


ROC Curve Example:


Churn Distribution Plot:


Key Insights

Class 1 (churned customers) is harder to predict → F1 lower.

Threshold tuning significantly improves F1-score for churned class.

Ensemble slightly improves ROC-AUC but CatBoost alone gives best balance.

Data imbalance in test set requires attention (SMOTE, class weighting).

Recommendations

Adjust thresholds per business objectives (maximize recall for churned customers).

Feature engineering: create derived features (tenure × monthly charges, etc.).

Ensemble stacking could boost performance further.

Document assumptions and hyperparameters for reproducibility.

Dependencies

Python 3.8+

pandas, numpy

scikit-learn

xgboost, lightgbm, catboost

matplotlib, seaborn (for EDA)

optuna (for hyperparameter tuning)

License

Educational and portfolio purposes.
