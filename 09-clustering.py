# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt


dataset = pd.read_csv("datasets/musteriler.csv")

print(dataset)
print(dataset.isnull().sum())

#%% Data Preprocessing
X = dataset.iloc[:,3:].values

#%% Clustering(K-Means)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, init="k-means++")
kmeans.fit(X)

print(kmeans.cluster_centers_)

#%% Optimize the number of cluster
results = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=123)
    kmeans.fit(X)
    results.append(kmeans.inertia_)

plt.plot(range(1, 11), results, "bs-")
plt.grid()
plt.show()

#%% Predictions
kmeans = KMeans(n_clusters=4, init="k-means++")

# kmeans.fit(X)
Y_pred = kmeans.fit_predict(X)

#print(kmeans.labels_)
print(Y_pred)

# plt.scatter(X[Y_pred==0, 0], X[Y_pred==0, 1], s=100, c="red")
# plt.scatter(X[Y_pred==1, 0], X[Y_pred==1, 1], s=100, c="blue")
# plt.scatter(X[Y_pred==2, 0], X[Y_pred==2, 1], s=100, c="green")
# plt.scatter(X[Y_pred==3, 0], X[Y_pred==3, 1], s=100, c="yellow")
for i in range(4):
    plt.scatter(X[Y_pred==i, 0], X[Y_pred==i, 1], s=80)
    
plt.title("K-Means")
plt.show()

#%% Clustering(Hierarchical(Agglomerative) Clustering)
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")
Y_pred = ac.fit_predict(X)

print(Y_pred)

# plt.scatter(X[Y_pred==0, 0], X[Y_pred==0, 1], s=100, c="red")
# plt.scatter(X[Y_pred==1, 0], X[Y_pred==1, 1], s=100, c="blue")
# plt.scatter(X[Y_pred==2, 0], X[Y_pred==2, 1], s=100, c="green")
# plt.scatter(X[Y_pred==3, 0], X[Y_pred==3, 1], s=100, c="yellow")

for i in range(4):
    plt.scatter(X[Y_pred==i, 0], X[Y_pred==i, 1], s=80)
    
plt.title("AgglomerativeClustering")
plt.show()

#%% Dendrogram

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.show()

