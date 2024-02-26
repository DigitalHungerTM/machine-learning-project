# kmeans.py
import numpy as np
from numpy.linalg import norm
from collections import defaultdict

type Datapoint = tuple
type Datapoint_indexes = list[int]
type Data = list[Datapoint]
type Centroid = Datapoint
type Centroids = list[Centroid]

def euclidean_distance(a, b, axis=None):
    # calculates euclidean distance between 2 vectors, a and b.
    a = np.array(a)
    b = np.array(b)
    return norm(a-b, axis=axis)


class KMeansClusterer:
    def __init__(self, n_clusters, n_feats):
        """
        Clusterer object that fits on a dataset and labels and predicts labels.

        Usage:
        >>> kmc = KMeansClusterer(n_clusters=..., n_feats=...)
        predictions = kmc.fit_predict(data=..., labels=...)
        
        :param `n_clusters`: number of clusters that the clusterer should make
        
        :param `n_feats`: number of dimensions that a single datapoint has
        """
        self.n_clusters: int = n_clusters
        self.n_feats: int = n_feats
        self.data: Data = []
        self.centroids: Centroids = []
        # clusters are stored as list of datapoint indexes according to self.data
        # this is done to use self.data as a sort of look up table later on
        self.clusters: defaultdict[Centroid, Datapoint_indexes] = defaultdict(list)

    def initialize_clusters(self):
        """
        Initializes cluster centroids with the Forgy method
        """
        random_indexes: list[int] = list(np.random.choice(list(range(len(self.data))), size=self.n_clusters))
        self.centroids = list(map(lambda index: self.data[index], random_indexes))

    def assign(self):
        """
        Assigns every datapoint to the closest centroid
        """
        self.clusters = defaultdict(list) # empty the clusters
        for datapoint_index, datapoint in enumerate(self.data):
            
            closest_centroid = min(self.centroids, key=lambda centroid: euclidean_distance(centroid, datapoint))
            
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
        """
        Fits on data and assigns labels to clusters
        
        :param `data`: list of datapoints
        :param `labels`: list of labels
        """
        assert isinstance(data, list), 'data should be a list'
        assert isinstance(data[0], tuple), 'datapoints should be tuples'
        self.data = data
        self.initialize_clusters()

        for _ in range(100):
            self.assign()
            self.update()
            # we call update last so self.centroids doesn't have
            # the same centroids as self.clusters

        # alternative (hopefully faster)
        predictions = [10000]*len(labels) # every datapoint should be assigned to a centroid, so no `10000` should be left in the predictions
        for centroid in self.clusters:
            # get datapoint indexes for the cluster that corresponds to the centroid
            datapoint_indexes = self.clusters[centroid]
            # map indexes to datapoints, while retaining the indexes
            datapoints = [(i, self.data[i]) for i in datapoint_indexes]
            # take index of datapoint that has closest distance to centroid
            i, _ = min(datapoints, key=lambda datapoint: euclidean_distance(centroid, datapoint[1]))
            label = labels[i]
            for index in datapoint_indexes:
                predictions[index] = label            

        return predictions
    