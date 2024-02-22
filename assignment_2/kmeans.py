# kmeans.py
import numpy as np
from numpy.linalg import norm
from collections import defaultdict

def euclidean_distance(a, b, axis=None):
    # calculates euclidean distance between 2 vectors, a and b.
    return norm(a-b, axis=axis)


class KMeansClusterer:
    def __init__(self, n_clusters, n_feats):
        self.n_clusters = n_clusters
        self.n_feats = n_feats
        self.data = None
        self.centroids = []
        self.clusters = defaultdict(list)

    def initialize_clusters(self):
        """
        Initializes cluster centroids with the Forgy method
        """
        not_chosen_indexes = list(range(len(self.data)))
        for _ in range(self.n_clusters):
            random_data_point_index = np.random.choice(not_chosen_indexes)
            del not_chosen_indexes[random_data_point_index] # make sure the same point doesn't get chosen twice
            self.centroids.append(self.data[random_data_point_index])

    def assign(self):
        """
        Assigns every datapoint in `data` to the closest cluster
        """
        self.clusters = defaultdict(list)
        for datapoint in self.data:
            closest_centroid = sorted(self.centroids, key=lambda centroid: euclidean_distance(centroid, datapoint))[0]
            self.clusters[closest_centroid].append(datapoint)


    def update(self):
        """
        Moves every centroid to the mean of its assigned datapoints
        """
        self.centroids = [] # emtpy the old centroids
        for cluster in self.clusters:
            cluster_datapoints = self.clusters[cluster]
            # by the definition of mean, all features in the new centroid lie between 0 and 1
            # so we can round to the nearest integer to get valid features
            new_centroid_raw = np.mean(cluster_datapoints, axis=0) # this is not rounded
            new_centroid_rounded = np.rint(new_centroid_raw) # rounded to the nearest integer
            self.centroids.append(new_centroid_rounded)


    def fit_predict(self, data):
        self.data = data
        # Fit contains the loop where you will first call initialize_clusters()
        self.initialize_clusters()

        for _ in range(10):
            self.assign()
            self.update()

        # Then call assign() and update() iteratively for 100 iterations

        return self.clusters
    