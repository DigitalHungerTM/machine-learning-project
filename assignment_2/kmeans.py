# kmeans.py
import numpy as np
from numpy.linalg import norm
from collections import defaultdict

type Datapoint = tuple
type Datapoint_indexes = list[int]
type Data = list[Datapoint]
type Centroid = Datapoint
type Centroids = list[Centroid]
type Centroid_key = int

def euclidean_distance(a, b, axis=None):
    # calculates euclidean distance between 2 vectors, a and b.
    a = np.array(a)
    b = np.array(b)
    return norm(a-b, axis=axis)



class KMeansClusterer:
    def __init__(self, n_clusters, n_feats):
        self.n_clusters: int = n_clusters
        self.n_feats: int = n_feats
        self.data: Data = []
        self.centroids: Centroids = []
        self.clusters: defaultdict[Centroid, Datapoint_indexes] = defaultdict(list)

    def initialize_clusters(self):
        """
        Initializes cluster centroids with the Forgy method
        """
        not_chosen_indexes = list(range(len(self.data)))
        for _ in range(self.n_clusters):

            random_data_point_index: int = np.random.choice(not_chosen_indexes)
            del not_chosen_indexes[random_data_point_index] # make sure the same point doesn't get chosen twice
            
            centroid = self.data[random_data_point_index] # convert to tuple to use as dict key
            self.centroids.append(centroid)

    def assign(self):
        """
        Assigns every datapoint to the closest centroid
        """
        self.clusters = defaultdict(list) # empty the clusters
        for datapoint_index, datapoint in enumerate(self.data):
            
            closest_centroid = sorted(self.centroids, key=lambda centroid: euclidean_distance(centroid, datapoint))[0]
            
            self.clusters[closest_centroid].append(datapoint_index)

    def update(self):
        """
        Moves every centroid to the mean of its assigned datapoints
        """
        self.centroids = [] # emtpy the old centroids
        for cluster in self.clusters.values():
            cluster_datapoints = [self.data[i] for i in cluster]
            new_centroid = tuple(np.mean(cluster_datapoints, axis=0))
            self.centroids.append(new_centroid)


    def fit_predict(self, data: Data, labels: list[int]):
        assert isinstance(data, list), 'data should be a list'
        assert isinstance(data[0], tuple), 'datapoints should be tuples'
        self.data = data
        self.initialize_clusters()

        for _ in range(100):
            self.assign()
            self.update()

        # map centroids to actual labels
        centroid_labels = {}
        for centroid in self.centroids:
            closest_datapoint = sorted(self.data, key=lambda datapoint: euclidean_distance(centroid, datapoint))[0]
            closest_datapoint_index = self.data.index(closest_datapoint)
            centroid_label = labels[closest_datapoint_index]
            centroid_labels[centroid] = centroid_label
            
        # compile predictions
        predictions = [0]*len(labels)
        for centroid in self.centroids:
            datapoint_indexes = self.clusters[centroid]
            for index in datapoint_indexes:
                predictions[index] = centroid_labels[centroid]

        return predictions
    