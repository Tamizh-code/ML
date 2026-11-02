from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Load dataset
d = load_iris()
X = StandardScaler().fit_transform(d.data)

# Apply K-Means
model = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = model.fit_predict(X)

# Interpret results
print("Clusters:", y_pred)
print("Centroids:\n", model.cluster_centers_)
print("Cluster Counts:", np.bincount(y_pred))
# # Experiment 22: Unsupervised learning using PCA dimensionality reduction
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# iris = load_iris()
# X_scaled = StandardScaler().fit_transform(iris.data)
# pca = PCA(n_components=2)
# X_2d = pca.fit_transform(X_scaled)

# print("Explained variance ratio:", pca.explained_variance_ratio_)
# print("Total variance retained:", pca.explained_variance_ratio_.sum())
# plt.scatter(X_2d[:, 0], X_2d[:, 1], c=iris.target)
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title("Iris Dataset: PCA Projection")
# plt.show()
