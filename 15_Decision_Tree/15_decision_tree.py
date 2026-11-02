from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
d = load_iris()
X, y = d.data, d.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction and accuracy
y_pred = model.predict(X_test)
print("Prediction:", y_pred)
print("Accuracy:", model.score(X_test, y_test))
# # Experiment 15: Decision Tree classification
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt

# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(
#     iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
# )
# model = DecisionTreeClassifier(max_depth=3, random_state=42)
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, pred))
# print("Confusion matrix:\n", confusion_matrix(y_test, pred))
# plt.figure(figsize=(12, 7))
# plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names,
#           filled=True, rounded=True)
# plt.title("Decision Tree")
# plt.show()
