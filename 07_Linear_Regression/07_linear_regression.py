# Experiment 7: Simple linear regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Synthetic relationship: study hours approximately determine exam marks.
rng = np.random.RandomState(42)
X = np.arange(1, 11).reshape(-1, 1)
y = 8 * X.ravel() + 25 + rng.normal(0, 5, size=10)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print("Slope:", model.coef_[0], "Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, pred), "R2:", r2_score(y_test, pred))
plt.scatter(X, y, label="Observed data")
line_x = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.plot(line_x, model.predict(line_x), label="Regression line")
plt.xlabel("Study hours")
plt.ylabel("Exam marks")
plt.title("Linear Regression")
plt.legend()
plt.show()
