# Financial Fraud Detection Pipeline
## Comprehensive Model Comparison & Artifact Bundling
### By Muhammad Auffa Hakim Aditya



This project presents a robust Machine Learning pipeline designed to detect fraudulent financial transactions. Using a synthetic financial dataset from Kaggle, the project evaluates four different classification algorithms and bundles all preprocessing objects and models into a single, highly portable deployment artifact.

The project was developed by Muhammad Auffa Hakim Aditya to demonstrate structured end-to-end Machine Learning engineering, moving from raw data processing to inference testing with a focus on clean deployment practices.

------------------------------------------------------------

PROJECT OBJECTIVES

1. Automatically fetch the Synthetic Financial Fraud dataset from Kaggle.
2. Implement efficient categorical encoding using Scikit-Learn's `OrdinalEncoder`, specifically configured to safely handle unknown categories during inference.
3. Standardize feature scales across the dataset using `MinMaxScaler`.
4. Train and compare the classification accuracy of 4 Machine Learning models:
   - Random Forest (Primary Model)
   - Logistic Regression
   - Support Vector Machine (SVM)
   - XGBoost
5. Bundle all models, scalers, encoders, and feature column lists into a single Python dictionary for unified serialization.
6. Simulate live production inference by extracting the saved objects and predicting the fraud status of a newly generated transaction record.

------------------------------------------------------------

DATASET INFORMATION

Source          : Kaggle (sriharshaeedala/financial-fraud-detection-dataset)
Domain          : Finance / Cybersecurity
Target Variable : isFraud (Binary: 0 = Legitimate, 1 = Fraud)

Features Analyzed:
- step: Maps a unit of time in the real world.
- type: Transaction type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER).
- amount: Transaction amount in local currency.
- nameOrig / nameDest: Origin and Destination account identifiers.
- oldbalanceOrg / newbalanceOrig: Initial and new balance before and after the transaction.
- oldbalanceDest / newbalanceDest: Initial and new balance of the recipient.
- isFlaggedFraud: System flag for massive transfers.

------------------------------------------------------------

MACHINE LEARNING PIPELINE ARCHITECTURE

1. Data Splitting: 80/20 train-test split using stratification to ensure the highly imbalanced fraud cases are proportionally represented in both sets.
2. Preprocessing:
   - `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`: Safely encodes transaction types and account IDs, ensuring the model doesn't crash if it encounters a completely new account ID in the future.
   - `MinMaxScaler`: Normalizes large currency amounts and balances to a standard scale.
3. Model Training & Evaluation:
   - Models are evaluated using Accuracy, Precision, Recall, and F1-Scores via comprehensive classification reports.

------------------------------------------------------------

MODEL SAVING & INFERENCE DEPLOYMENT

Unlike traditional workflows that save multiple `.pkl` files, this project packages the entire environment into a single dictionary artifact. 

Exported File:
- saved_models/fraud_detection_pipeline.pkl 

Contained Artifacts:
- "encoder": The trained OrdinalEncoder
- "scaler": The trained MinMaxScaler
- "random_forest_model": Trained RF Classifier
- "logistic_regression_model": Trained LR Classifier
- "svm_model": Trained SVC Classifier
- "xgboost_model": Trained XGB Classifier
- "feature_columns": List of strict feature inputs
- "categorical_columns": List of categorical features

------------------------------------------------------------

INSTALLATION

Install the required dependencies:

pip install pandas scikit-learn xgboost kagglehub joblib

------------------------------------------------------------

HOW TO RUN

1. Clone this repository:
   git clone https://github.com/YOUR_USERNAME/financial-fraud-detection.git

2. Install the required libraries.
3. Run the Python script. The script will download the dataset, train all models, evaluate them, bundle everything into `fraud_detection_pipeline.pkl`, and run a mock inference test on a $1,000 TRANSFER transaction.

------------------------------------------------------------

AUTHOR

Muhammad Auffa Hakim Aditya

This project was developed as an exploration of:
- Financial Fraud Detection
- End-to-End Supervised Machine Learning
- Advanced Categorical Encoding (Handling Unknowns)
- Unified Model Serialization (Artifact Bundling)
- Inference Simulation

------------------------------------------------------------

KEYWORDS 

- Muhammad Auffa Hakim Aditya
- Fraud Detection ML
- Scikit-Learn Pipeline
- XGBoost Classification
- Random Forest
- ML Deployment Artifacts
- ML Portfolio Project

------------------------------------------------------------

Note:
This project utilizes a synthetic dataset intended for educational, research, and algorithmic testing purposes.
