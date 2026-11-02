from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load dataset
d = load_iris()
X, y = d.data, d.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = SVC()
model.fit(X_train, y_train)

# Prediction and accuracy
y_pred = model.predict(X_test)
print("Prediction:", y_pred)
print("Accuracy:", model.score(X_test, y_test))
# # Experiment 14: Support Vector Machine classification
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# data = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#     data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
# )
# model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, pred))
# print("Confusion matrix:\n", confusion_matrix(y_test, pred))
# print(classification_report(y_test, pred, target_names=data.target_names))
