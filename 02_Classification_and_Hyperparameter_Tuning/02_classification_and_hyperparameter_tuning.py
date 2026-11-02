from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
d = load_iris()
X, y = d.data, d.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create model
model = DecisionTreeClassifier()

# Hyperparameter tuning
grid = GridSearchCV(model, {'max_depth': [2, 3, 4, 5]}, cv=5)
grid.fit(X_train, y_train)

# Prediction
y_pred = grid.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Best Parameters:", grid.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# # Experiment 2: Classification model evaluation and hyperparameter tuning
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# X, y = load_breast_cancer(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
# pipe = Pipeline([("scale", StandardScaler()), ("model", SVC())])
# params = {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"],
#           "model__gamma": ["scale", "auto"]}
# search = GridSearchCV(pipe, params, cv=5, scoring="accuracy", n_jobs=-1)
# search.fit(X_train, y_train)
# pred = search.predict(X_test)
# print("Best parameters:", search.best_params_)
# print("Accuracy:", accuracy_score(y_test, pred))
# print("Confusion matrix:\n", confusion_matrix(y_test, pred))
# print(classification_report(y_test, pred, target_names=load_breast_cancer().target_names))
