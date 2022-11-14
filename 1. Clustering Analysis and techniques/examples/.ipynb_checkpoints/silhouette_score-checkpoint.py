## This python script demonstrates the use of the silhouette score method.
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.datasets import load_iris

dataset = load_iris()
X = dataset.data
y = dataset.target

kmeans_model = KMeans(n_clusters = 3, random_state = 1).fit(X)
labels = kmeans_model.labels_

print(silhouette_score(X, labels, metric = 'euclidean'))