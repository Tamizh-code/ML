from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
d = load_iris()
X, y = d.data, d.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n",
      classification_report(y_test, y_pred))
# # Experiment 6: Naive Bayes accuracy, confusion matrix, and classification report
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(
#     iris.data, iris.target, test_size=0.25, random_state=7, stratify=iris.target
# )
# model = GaussianNB()
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, pred))
# print("Confusion matrix:\n", confusion_matrix(y_test, pred))
# print("Classification report:\n", classification_report(
#     y_test, pred, target_names=iris.target_names
# ))
