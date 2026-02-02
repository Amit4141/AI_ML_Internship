Task 11: SVM Breast Cancer Classification

Project Title:
SVM Based Breast Cancer Classification using Scikit-learn

Description:
This project implements Support Vector Machine (SVM) for classifying breast cancer tumors as malignant or benign. The project focuses on kernel-based classification, feature scaling, hyperparameter tuning, and model evaluation using ROC curve and AUC score.

Tools and Technologies:
Python
Scikit-learn
Matplotlib
Joblib
Jupyter Notebook

Dataset Information:
Dataset Name: Breast Cancer Wisconsin Dataset
Source: sklearn.datasets.load_breast_cancer
Total Samples: 569
Total Features: 30
Target Classes:
0 – Malignant
1 – Benign

Project Steps:

Load the breast cancer dataset and inspect feature and label distribution.
Apply StandardScaler to normalize feature values.
Split the dataset into training and testing sets.
Train a baseline SVM model using a linear kernel.	
Train an SVM model using the RBF kernel and compare accuracy.
Tune hyperparameters C and gamma using GridSearchCV.
Evaluate the best model using confusion matrix and classification report.
Plot ROC curve and calculate AUC score.
Save the trained model pipeline (scaler + SVM) for reuse.

Model Details:
Algorithm: Support Vector Machine (SVM)
Kernels Used: Linear, RBF
Hyperparameters Tuned:
C – Regularization parameter
Gamma – Kernel coefficient

Evaluation Metrics:
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
ROC Curve
AUC Score

Files Included:
svm_breast_cancer.ipynb – Jupyter Notebook containing complete code
svm_breast_cancer_model.pkl – Saved trained SVM model
README.txt – Project documentation

How to Run:
Open Jupyter Notebook.
Open svm_breast_cancer.ipynb.
Run all cells in order.
View ROC curve and AUC score output.
The trained model will be saved automatically.

How to Load Saved Model:
loaded_model = joblib.load("svm_breast_cancer_model.pkl")

Final Outcome:
This task helps the intern understand SVM classification, kernel selection, feature scaling, hyperparameter tuning, and model evaluation techniques used in real-world machine learning problems.