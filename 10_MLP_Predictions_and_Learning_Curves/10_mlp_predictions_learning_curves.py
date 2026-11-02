# Experiment 10: MLP predictions and loss/accuracy learning plots
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(20,), solver="adam", learning_rate_init=0.01,
                      max_iter=1, warm_start=True, random_state=42)
losses, accuracies = [], []
for epoch in range(100):
    model.fit(X_train, y_train)
    losses.append(model.loss_)
    accuracies.append(accuracy_score(y_test, model.predict(X_test)))

print("Test accuracy:", accuracies[-1])
sample = scaler.transform([[5.1, 3.5, 1.4, 0.2]])
print("New sample prediction:", iris.target_names[model.predict(sample)[0]])

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Training Loss")
plt.show()
plt.plot(accuracies)
plt.xlabel("Epoch")
plt.ylabel("Test accuracy")
plt.title("MLP Accuracy During Training")
plt.show()
