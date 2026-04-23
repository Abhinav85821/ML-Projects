# =========================
# SVM MINI PROJECT
# Breast Cancer Classification
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -------------------------
# 1. Load Dataset
# -------------------------
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset Loaded Successfully")
print("Shape:", X.shape)

# -------------------------
# 2. Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 3. Feature Scaling
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# 4. Train SVM Model
# -------------------------
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# -------------------------
# 5. Predictions
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# 6. Evaluation
# -------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------
# 7. Save Model
# -------------------------
joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved successfully as svm_model.pkl")
print("Scaler saved successfully as scaler.pkl")

# -------------------------
# 8. Sample Prediction
# -------------------------
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)

print("\nSample Prediction:")
print("Prediction (0 = Malignant, 1 = Benign):", prediction[0])