import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([20000, 30000, 40000, 50000, 60000])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict new values
X_new = np.array([[6], [7]])
y_pred = model.predict(X_new)

print("Predicted Salary:", y_pred)

# Regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.scatter(X_new, y_pred, color='green')

plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Linear Regression")
plt.show()
# # Experiment 8: Predict new values and plot the regression line
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.linear_model import LinearRegression

# X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
# y = np.array([35, 40, 48, 52, 60, 65, 70, 78])
# model = LinearRegression().fit(X, y)

# new_hours = np.array([[2.5], [5.5], [9]])
# print("Predictions for study hours", new_hours.ravel(), ":", model.predict(new_hours))
# line_x = np.linspace(1, 9, 100).reshape(-1, 1)
# plt.scatter(X, y, color="blue", label="Training data")
# plt.plot(line_x, model.predict(line_x), color="red", label="Regression line")
# plt.scatter(new_hours, model.predict(new_hours), color="green", label="New predictions")
# plt.xlabel("Study hours")
# plt.ylabel("Marks")
# plt.title("Linear Regression Predictions")
# plt.legend()
# plt.show()
