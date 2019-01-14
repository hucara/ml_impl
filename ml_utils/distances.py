import numpy as np


class Distances:

    @staticmethod
    def euclidean_dist(a, b):
        return np.linalg.norm(a - b)

    @staticmethod
    def mahalanobis_dist(a, b):
        e = a - b
        x = np.vstack([a, b])
        v = np.cov(x.T)
        p = np.linalg.inv(v)
        d = np.sqrt(np.sum(np.dot(e, p) * e, axis=1))
        return d

