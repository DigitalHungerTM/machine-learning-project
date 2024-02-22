# kmeans.py
import numpy as np
from numpy.linalg import norm

def euclidean_distance(a, b, axis=None):
    # calculates euclidean distance between 2 vectors, a and b.
    return norm(a-b, axis=axis)


class KMeansClusterer:
    def __init__(self, n_clusters, n_feats):
        self.n_clusters = n_clusters
        self.n_feats = n_feats
        self.centroids = np.zeros((n_clusters, n_feats))

    def initialize_clusters(self, data):
        # initalize your clusters centers here with the Forgy method, i.e. 
        # by assigning each cluster center to a random point in your data.
        pass

    def assign(self, data):
        # in this function, you need to assign your data points to the nearest cluster center
        pass

    def update(self, data, assignments):
        # in this function, you need to update your cluster centers, according to the mean of 
        # the points in that cluster
        pass

    def fit_predict(self, data):
        # Fit contains the loop where you will first call initialize_clusters()

        # Then call assign() and update() iteratively for 100 iterations

        # Return the assignments
        pass