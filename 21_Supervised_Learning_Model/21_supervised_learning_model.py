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

# Prediction
y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", model.score(X_test, y_test))
# # Experiment 21: General supervised learning workflow (classification)
# from sklearn.datasets import load_wine
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report

# data = load_wine()
# X_train, X_test, y_train, y_test = train_test_split(
#     data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
# )
# model = make_pipeline(
#     SimpleImputer(strategy="median"),
#     StandardScaler(),
#     RandomForestClassifier(n_estimators=150, random_state=42)
# )
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, pred))
# print(classification_report(y_test, pred, target_names=data.target_names))
