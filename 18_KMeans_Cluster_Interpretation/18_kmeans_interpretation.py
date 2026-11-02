import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

d = load_iris()
X = StandardScaler().fit_transform(d.data)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = model.fit_predict(X)

print("Centroids:\n", model.cluster_centers_)
print("Cluster Counts:", np.bincount(y_pred))

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("K-Means Clusters")
plt.show()
# # Experiment 18: Interpret K-means clusters, centroids, and cluster distributions
# import pandas as pd
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans

# iris = load_iris(as_frame=True)
# X = iris.data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# model = KMeans(n_clusters=3, random_state=42, n_init=10)
# labels = model.fit_predict(X_scaled)

# result = X.copy()
# result["cluster"] = labels
# print("Number of points in each cluster:\n", result["cluster"].value_counts().sort_index())
# print("\nMean feature values by cluster:\n", result.groupby("cluster")[X.columns].mean())
# centers_original_scale = scaler.inverse_transform(model.cluster_centers_)
# print("\nCluster centroids in original feature units:\n",
#       pd.DataFrame(centers_original_scale, columns=X.columns))
# print("\nInterpretation: compare each cluster's mean feature values to identify groups.")
