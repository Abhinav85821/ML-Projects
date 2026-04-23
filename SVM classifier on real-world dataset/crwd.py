# ==========================================
# SVM CLASSIFIER (Single Complete Code)
# Dataset: Breast Cancer Wisconsin (sklearn)
# Kernels: Linear, RBF, Polynomial
# Includes: Scaling, CV, GridSearch, Evaluation
# ==========================================

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# Load dataset
# -----------------------------
data = load_breast_cancer()
X = data.data
y = data.target

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------
# Feature scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Function to evaluate model
# -----------------------------
def evaluate(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n==============================")
    print(name)
    print("==============================")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 1. SVM Kernels Comparison
# -----------------------------
models = {
    "Linear SVM": SVC(kernel='linear', C=1),
    "RBF SVM": SVC(kernel='rbf', C=1, gamma='scale'),
    "Polynomial SVM": SVC(kernel='poly', C=1, degree=3)
}

for name, model in models.items():
    evaluate(model, name)

# -----------------------------
# 2. Cross Validation
# -----------------------------
print("\nCross Validation Scores:")
for k in ['linear', 'rbf', 'poly']:
    model = SVC(kernel=k, C=1)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{k}: {scores.mean():.4f}")

# -----------------------------
# 3. Hyperparameter Tuning (Best Model)
# -----------------------------
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# -----------------------------
# Final Evaluation
# -----------------------------
print("\nFINAL OPTIMIZED MODEL RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))