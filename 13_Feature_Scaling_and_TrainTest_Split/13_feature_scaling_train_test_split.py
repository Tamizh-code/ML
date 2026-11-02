# Experiment 13: Feature scaling and train/test splitting
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Original training shape:", X_train.shape)
print("Scaled training shape:", X_train_scaled.shape)
print("Original testing shape:", X_test.shape)
print("Scaled testing shape:", X_test_scaled.shape)
print("Training mean of first scaled feature:", X_train_scaled[:, 0].mean())
