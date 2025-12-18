# Predict-Customer-Churn-Ml
# Customer Churn Prediction

## Overview
This project implements a complete pipeline for predicting customer churn using **Machine Learning**.  
It combines multiple models, hyperparameter optimization, and threshold tuning to maximize **F1-score** and **ROC-AUC**.  

The code is organized for **production-level usage**, modular and maintainable.

**Key Features:**
- Data preprocessing and feature engineering
- Model training and evaluation for **CatBoost**, **XGBoost**, and **LightGBM**
- Threshold optimization for improved F1-score
- Ensemble evaluation for robust performance
- Configurable for different datasets
- Ready for deployment or further integration

---

## Project Structure

customer_churn_ml/
│
├── data/ # Raw and processed datasets
│ ├── train.csv
│ ├── test.csv
│
├── src/ # Source code
│ ├── preprocessing.py # Data preprocessing and feature engineering
│ ├── train.py # Training pipeline
│ ├── evaluate.py # Evaluation scripts
│ └── utils.py # Utility functions
│
├── models/ # Saved models and serialized objects
│ ├── catboost_model.pkl
│ ├── xgboost_model.pkl
│
├── notebooks/ # Jupyter notebooks for experiments
│ └── EDA_and_training.ipynb
│
├── configs/ # Configuration files
│ └── config.yaml
│
├── requirements.txt # Python dependencies
├── README.md
└── .gitignore
