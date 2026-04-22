import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# Load dataset (fixed path issue)
# -------------------------------
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, "data.csv")

data = pd.read_csv(file_path)

# -------------------------------
# Prepare data
# -------------------------------
X = data[['hours', 'attendance']]  # features
y = data['pass']                   # target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train model
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# Test model
# -------------------------------
y_pred = model.predict(X_test)

print("\n📊 Model Accuracy:", accuracy_score(y_test, y_pred))

# -------------------------------
# User input prediction
# -------------------------------
print("\n🔍 Enter student details for prediction:")

try:
    hours = float(input("Enter study hours: "))
    attendance = float(input("Enter attendance (%): "))

    prediction = model.predict([[hours, attendance]])

    if prediction[0] == 1:
        print("✅ Prediction: Student will PASS")
    else:
        print("❌ Prediction: Student will FAIL")

except ValueError:
    print("⚠️ Please enter valid numeric values!")

# -------------------------------
# Optional: Show model coefficients
# -------------------------------
print("\n📈 Model Details:")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)