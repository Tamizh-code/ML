from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
d = load_iris()
X, y = d.data, d.target

# Normalize features
X = StandardScaler().fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Display output
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
# # Experiment 12: KNN preprocessing, normalization, and train/test split
# from sklearn.datasets import load_wine
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score

# wine = load_wine()
# X_train, X_test, y_train, y_test = train_test_split(
#     wine.data, wine.target, test_size=0.25, random_state=42, stratify=wine.target
# )
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
# print("Training samples:", len(X_train), "Testing samples:", len(X_test))
# print("KNN accuracy:", accuracy_score(y_test, model.predict(X_test)))
