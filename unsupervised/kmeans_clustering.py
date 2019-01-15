from abs_model import AbsModel
from ml_utils.distances import Distances

import random
import numpy as np


class KMeans(AbsModel):

    def __init__(self, k=3, max_iter = 25, metric="euclidean"):
        if metric == "euclidean":
            self._metric = Distances.euclidean_dist
        elif metric == "mahalanobis":
            self._metric = Distances.mahalanobis_dist
        elif callable(metric):
            self._metric = metric

        self._centroids = None
        self._k = k
        self._X = None
        self._distances = self._neighbors = None
        self._max_iter = max_iter

    def _init_centroids(self):
        self._centroids = []
        for i in range(self._k):
            self._centroids.append(random.choice(self._X))

    def train(self, X):
        self._X = X
        self._init_centroids()

    def predict(self):
        assigned_centroids = np.zeros(len(self._X))
        converged = np.zeros(self._k)

        for i in range(self._max_iter):
            old_centroids = self._centroids.copy()

            # Expectation: assign each data point to closest cluster
            for j, d in enumerate(self._X):
                dist_to_centroids = [self._metric(d, c) for c in self._centroids]
                assigned_centroids[j] = np.argmin(dist_to_centroids)

            # Maximization: compute new centroid location as a mean of the cluster
            for j in range(self._k):
                # get mean of all centroid[j] data and compute a new centroid mean
                centroid_data = [self._X[d] for d in range(len(self._X)) if assigned_centroids[d] == j]
                self._centroids[j] = np.mean(centroid_data, axis=0)

                converged[j] = np.array_equal(old_centroids[j], self._centroids[j])

            if all(converged):
                print("Converged in iteration {}.".format(i))
                return (assigned_centroids, self._centroids)

        return (assigned_centroids, self._centroids)
