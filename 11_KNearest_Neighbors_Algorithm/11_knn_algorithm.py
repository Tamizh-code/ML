from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
d = load_iris()
X, y = d.data, d.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Prediction and accuracy
y_pred = model.predict(X_test)
print("Prediction:", y_pred)
print("Accuracy:", model.score(X_test, y_test))
# # Experiment 11: K-Nearest Neighbors classification
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, classification_report

# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(
#     iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
# )
# model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, pred))
# print(classification_report(y_test, pred, target_names=iris.target_names))
