import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed


def dist_compute(X, centroids):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((X[i] - centroids) ** 2, axis=0))
    return dis_mat


class CentroidManager(object):

    def __init__(self, feature_dim, n_clusters):
        self.feature_dim = feature_dim
        self.n_clusters = n_clusters
        self.centroids = np.zeros((self.n_clusters, self.feature_dim))
        self.count = 100 * np.ones((self.n_clusters))
        self.n_jobs = 1

    def parallel_dist_compute(self, X):
        dist = Parallel(n_jobs=self.n_jobs)(
            delayed(dist_compute)(X, self.centroids[i])
            for i in range(self.n_clusters))
        dist = np.hstack(dist)
        return dist

    def init_clusters(self, X):
        model = KMeans(n_clusters=self.n_clusters, n_init=20)
        model.fit(X)
        self.centroids = model.cluster_centers_

    def update_cluster(self, X, cluster_id):
        n_samples = X.shape[0]
        for i in range(n_samples):
            self.count[cluster_id] += 1
            eta = 1.0 / self.count[cluster_id]
            updated_cluster = ((1 - eta) * self.centroids[cluster_id] +
                               eta * X[i])
            self.centroids[cluster_id] = updated_cluster
    
    def update_assingment(self, X):
        dist = self.parallel_dist_compute(X)
        return np.argmin(dist, axis=1)
