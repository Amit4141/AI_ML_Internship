CREDIT CARD FRAUD DETECTION USING RANDOM FOREST

--------------------------------------------------
PROJECT OVERVIEW
--------------------------------------------------
This project focuses on detecting fraudulent credit card transactions using
machine learning techniques. Due to the highly imbalanced nature of fraud data,
traditional accuracy metrics are not reliable. Hence, this project emphasizes
precision, recall, and F1-score for evaluation.

The Random Forest algorithm is used as the main model to understand ensemble
learning and handle imbalanced datasets effectively.

--------------------------------------------------
DATASET INFORMATION
--------------------------------------------------
Dataset Name:
Credit Card Fraud Detection Dataset

Source:
Kaggle – https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

File Name:
creditcard.csv

Total Records:
284,807 transactions

Fraud Cases:
492 transactions

Target Column:
Class
0 = Legitimate Transaction
1 = Fraudulent Transaction

--------------------------------------------------
FEATURE DESCRIPTION
--------------------------------------------------
Time   : Seconds elapsed between transactions
V1-V28: PCA transformed numerical features
Amount: Transaction amount
Class : Target variable (0 or 1)

Note:
V1 to V28 are anonymized features and do not require interpretation.

--------------------------------------------------
TOOLS AND TECHNOLOGIES USED
--------------------------------------------------
Programming Language:
Python

Libraries:
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

Algorithms:
- Logistic Regression (Baseline Model)
- Random Forest Classifier

--------------------------------------------------
PROJECT WORKFLOW
--------------------------------------------------
1. Load the dataset
2. Analyze class imbalance
3. Separate features and target
4. Perform stratified train-test split
5. Train baseline Logistic Regression model
6. Train Random Forest model
7. Evaluate models using Precision, Recall, F1-score
8. Plot feature importance
9. Save trained model for future use

--------------------------------------------------
MODEL EVALUATION METRICS
--------------------------------------------------
Accuracy is not used as the main metric due to class imbalance.

Evaluation Metrics Used:
- Precision
- Recall
- F1-score

These metrics better reflect fraud detection performance.

--------------------------------------------------
FEATURE IMPORTANCE
--------------------------------------------------
Random Forest provides feature importance scores that help identify
the most influential variables contributing to fraud detection.

Top features are visualized using a bar graph.

--------------------------------------------------
MODEL SAVING
--------------------------------------------------
The trained Random Forest model is saved using Joblib.

Saved File:
fraud_detection_rf_model.pkl

This model can be reused for future predictions.

--------------------------------------------------
FINAL OUTCOME
--------------------------------------------------
Through this project, the following concepts are learned:
- Handling imbalanced datasets
- Ensemble learning with Random Forest
- Importance of proper evaluation metrics
- Real-world fraud detection workflow

--------------------------------------------------
END OF README
--------------------------------------------------
