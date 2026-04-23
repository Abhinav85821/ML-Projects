# Digit Recognition System using SVM (Single Code)

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def main():
    # Load dataset
    digits = datasets.load_digits()

    # Features and labels
    X = digits.data
    y = digits.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # SVM model
    model = SVC(kernel='rbf', gamma=0.001, C=10)

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Show some predictions
    plt.figure(figsize=(10, 4))

    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
        plt.title(f"P:{model.predict([X_test[i]])[0]}")
        plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()