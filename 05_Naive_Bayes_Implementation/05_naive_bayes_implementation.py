# Experiment 5: Naive Bayes classification with preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
)
model = make_pipeline(StandardScaler(), GaussianNB())
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Naive Bayes accuracy:", accuracy_score(y_test, pred))
print("First five predictions:", iris.target_names[pred[:5]])
