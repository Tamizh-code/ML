# Experiment 8: Predict new values and plot the regression line
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([35, 40, 48, 52, 60, 65, 70, 78])
model = LinearRegression().fit(X, y)

new_hours = np.array([[2.5], [5.5], [9]])
print("Predictions for study hours", new_hours.ravel(), ":", model.predict(new_hours))
line_x = np.linspace(1, 9, 100).reshape(-1, 1)
plt.scatter(X, y, color="blue", label="Training data")
plt.plot(line_x, model.predict(line_x), color="red", label="Regression line")
plt.scatter(new_hours, model.predict(new_hours), color="green", label="New predictions")
plt.xlabel("Study hours")
plt.ylabel("Marks")
plt.title("Linear Regression Predictions")
plt.legend()
plt.show()
