from abs_model import AbsModel
from collections import Counter
from ml_utils.distances import Distances


class KnnClassifier(AbsModel):

    def __init__(self, k=3, metric="euclidean"):
        if metric == "euclidean":
            self._metric = Distances.euclidean_dist
        elif metric == "mahalanobis":
            self._metric = Distances.mahalanobis_dist
        elif callable(metric):
            self._metric = metric

        self._k = k
        self._X = self._y = None
        self._distances = self._neighbors = None

    def train(self, X, y):
        self._X = X
        self._y = y

    def predict(self, test_x):
        self._distances = []

        # get distances between train set and new point
        for i in range(len(self._X)):
            self._distances.append(self._metric(self._X[i], test_x))

        # get k closest distances
        self._neighbors = sorted(self._distances)[:self._k]

        # get label of those neighbors
        labels = []
        for i in range(len(self._neighbors)):
            idx = self._distances.index(self._neighbors[i])
            labels.append(self._y[idx])

        # vote class
        return Counter(labels).most_common(1)[0][0]

