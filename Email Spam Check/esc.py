import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load dataset (ensure your CSV has 'text' and 'label' columns)
df = pd.read_csv("emails.csv")

# Label Encoding (spam/ham → 1/0)
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

X = df['text']
y = df['label']

# Pipeline: TF-IDF + Logistic Regression
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('lr', LogisticRegression(max_iter=1000))
])

# K-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print("Cross-validation scores:", scores)
print("Mean Accuracy:", scores.mean())

# Train final model
model.fit(X, y)

# Test prediction
sample_email = ["Congratulations! You have won a free lottery ticket"]
prediction = model.predict(sample_email)

print("Prediction:", le.inverse_transform(prediction))