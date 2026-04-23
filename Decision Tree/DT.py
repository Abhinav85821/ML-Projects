# Decision Tree Classifier Mini Project (Single File)
# Dataset: Iris

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import joblib

# -----------------------------
# 1. Load Dataset
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# -----------------------------
# 2. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Train Decision Tree Model
# -----------------------------
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 4. Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 5. Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("🌳 Decision Tree Results")
print("==============================")
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# 6. Visualize Decision Tree
# -----------------------------
plt.figure(figsize=(15,10))
plot_tree(
    model,
    feature_names=feature_names,
    class_names=class_names,
    filled=True
)
plt.title("Decision Tree Visualization")
plt.show()

# -----------------------------
# 7. Save Model
# -----------------------------
joblib.dump(model, "decision_tree_model.pkl")
print("\n✅ Model saved as decision_tree_model.pkl")

# -----------------------------
# 8. Test on Custom Input
# -----------------------------
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample)

print("\n🔮 Prediction for sample input:", class_names[prediction[0]])