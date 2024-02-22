# kmeans.py
import numpy as np
from numpy.linalg import norm
from collections import defaultdict

def euclidean_distance(a, b, axis=None):
    # calculates euclidean distance between 2 vectors, a and b.
    return norm(a-b, axis=axis)

type datapoint = list[int]
type data = list[datapoint]
type centroid_key = int


class KMeansClusterer:
    def __init__(self, n_clusters, n_feats):
        self.n_clusters: int = n_clusters
        self.n_feats: int = n_feats
        self.data: data = []
        self.centroids: list[tuple[centroid_key, datapoint]] = []
        self.clusters: defaultdict[centroid_key, data] = defaultdict(list)

    def initialize_clusters(self):
        """
        Initializes cluster centroids with the Forgy method
        """
        not_chosen_indexes = list(range(len(self.data)))
        for i in range(self.n_clusters):
            random_data_point_index: int = np.random.choice(not_chosen_indexes)
            del not_chosen_indexes[random_data_point_index] # make sure the same point doesn't get chosen twice
            centroid = self.data[random_data_point_index]
            self.centroids.append((i, centroid)) # use i as arbitrary (but unique) label

    def assign(self):
        """
        Assigns every datapoint in `data` to the closest cluster
        """
        self.clusters = defaultdict(list)
        for datapoint in self.data:
            # get arbitrary label for the closest centroid
            i, closest_centroid = sorted(self.centroids, key=lambda centroid: euclidean_distance(centroid[1], datapoint))[0]
            # store the datapoint in the corresponding cluster
            self.clusters[i].append(datapoint)


    def update(self):
        """
        Moves every centroid to the mean of its assigned datapoints
        """
        self.centroids = [] # emtpy the old centroids
        for i in self.clusters:
            cluster = self.clusters[i]
            # by the definition of mean, all features in the new centroid lie between 0 and 1
            # so we can round to the nearest integer to get valid features
            new_centroid_raw = np.mean(cluster, axis=0) # this is not rounded
            new_centroid_rounded = np.rint(new_centroid_raw) # rounded to the nearest integer
            self.centroids.append((i, new_centroid_rounded))


    def fit_predict(self, data: list[list[int]]):
        self.data = data
        # Fit contains the loop where you will first call initialize_clusters()
        self.initialize_clusters()

        for _ in range(10):
            self.assign()
            self.update()

        # Then call assign() and update() iteratively for 100 iterations

        return self.clusters
    