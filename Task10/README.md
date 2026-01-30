# Task 10: KNN – Handwritten Digit Classification

## Project Overview

This project demonstrates **Handwritten Digit Classification** using the **K-Nearest Neighbors (KNN)** algorithm. The goal is to understand **distance-based classification**, feature scaling, and tuning the value of **K** for better accuracy.

The project uses the **Sklearn Digits Dataset**, which contains handwritten digits from 0 to 9.

---

## Dataset Information

### Primary Dataset: Sklearn Digits Dataset

* Source: Scikit-learn built-in dataset
* Total Samples: 1797
* Image Size: 8 × 8 pixels (grayscale)
* Classes: Digits 0 to 9

### Dataset Loading Code

```python
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target
```

No manual download is required.

### Alternative Dataset: MNIST (Optional)

* 60,000 training images
* 10,000 test images
* Image Size: 28 × 28 pixels

Official Link:
[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

---

## Tools & Technologies Used

* Python
* Scikit-learn
* Matplotlib
* Jupyter Notebook

---

## Project Workflow

1. Load the digits dataset
2. Check shape of features (X) and labels (y)
3. Visualize sample digit images
4. Split dataset into training and testing sets
5. Apply feature scaling using StandardScaler
6. Train KNN model with K = 3
7. Evaluate accuracy
8. Try multiple K values (3, 5, 7, 9)
9. Plot Accuracy vs K graph
10. Generate confusion matrix
11. Display test images with predicted labels

---

## Feature Scaling

KNN uses distance-based calculations, so feature scaling is required.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Model Training

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
```

---

## Accuracy Evaluation

Accuracy is calculated using:

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
```

Expected accuracy is around 97% to 98%.

---

## Accuracy vs K Plot

The best K value is chosen by comparing accuracy values for multiple K values.

K values tested:

* K = 3
* K = 5
* K = 7
* K = 9

---

## Confusion Matrix

A confusion matrix is generated to analyze misclassified digits.

```python
from sklearn.metrics import confusion_matrix
```

---

## Final Output Visualization

The model displays sample test images along with their predicted digit labels.

---

## Deliverables

* Jupyter Notebook
* Accuracy vs K graph
* Confusion Matrix
* Predicted digit output images

---

## Final Outcome

* Understanding of distance-based classification
* Importance of feature scaling in KNN
* Effect of K value on model accuracy
* Practical experience with handwritten digit recognition

---

## Conclusion

This project successfully implements the KNN algorithm for handwritten digit classification and achieves high accuracy using proper feature scaling and K tuning.

---

