# kmeans.py
import numpy as np
from numpy.linalg import norm
from collections import Counter, defaultdict

type Datapoint = tuple
type Datapoint_indexes = list[int]
type Data = list[Datapoint]
type Centroid = Datapoint
type Centroids = list[Centroid]

def euclidean_distance(a, b, axis=None):
    """
    Calculate Euclidean distance between vectors a and b

    :param `a`: ArrayLike, vector of size N
    :param `b`: ArrayLike, vector of size N
    :return: distance between vectors a and b 
    """
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

        predictions = [0]*len(labels) # every datapoint is assigned to a cluster, so every value in this list will be overwritten
        
        for centroid in self.clusters:
        
            # get datapoint indexes for the cluster that corresponds to the centroid
            datapoint_indexes = self.clusters[centroid]

            # take the label of the datapoint nearest to the centroid
            datapoints = [(i, self.data[i]) for i in datapoint_indexes]
            i, _ = min(datapoints, key=lambda datapoint: euclidean_distance(centroid, datapoint[1]))
            cluster_centroid_nearest_label = labels[i]

            # take most common label for datapoints in the cluster
            cluster_labels = [labels[i] for i in datapoint_indexes]
            cluster_labels_counter = Counter(cluster_labels)
            most_common_cluster_label = cluster_labels_counter.most_common(1)[0][0]

            for index in datapoint_indexes:
                predictions[index] = most_common_cluster_label

        return predictions
    