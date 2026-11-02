# Experiment 17: K-means clustering with feature scaling
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
print("Cluster labels:", labels)
print("Inertia:", kmeans.inertia_)
print("Silhouette score:", silhouette_score(X, labels))
print("Cluster centers (scaled features):\n", kmeans.cluster_centers_)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker="X", s=200, label="Centroids")
plt.xlabel("Scaled feature 1")
plt.ylabel("Scaled feature 2")
plt.title("K-means Clustering")
plt.legend()
plt.show()
