# kmeans.py
from typing import Literal
import numpy as np
from numpy.linalg import norm
from collections import Counter, defaultdict
import random


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
    def __init__(self, n_clusters):
        """
        Clusterer object that fits on a dataset and labels and predicts labels.

        Usage:
        >>> kmc = KMeansClusterer(n_clusters=...)
        predictions = kmc.fit_predict(data=..., labels=...)
        
        :param `n_clusters`: number of clusters that the clusterer should make
        """
        self.n_clusters: int = n_clusters
        self.data: list[tuple] = []
        self.centroids: list[tuple] = []
        # clusters are stored as list of datapoint indexes according to self.data
        # this is done to preserve self.data as a sort of look up table for later on
        self.clusters: defaultdict[tuple, list[int]] = defaultdict(list)

    def initialize_clusters(self):
        """
        Initializes cluster centroids with the Forgy method
        """
        # choose random unique datapoints as initial centroids
        self.centroids = random.sample(self.data, k=self.n_clusters)

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

    def fit(self, n: int):
        """
        Fits on data by iteratively calling `assign()` and `update()`

        :param `n`: number of iterations
        """
        self.initialize_clusters()
        for _ in range(n):
            self.assign()
            self.update()

    def predict(self, mode: Literal['nearest', 'most_common']='nearest') -> list[int]:
        """
        Predicts label on the currently fitted clusters

        :param `mode`: mode by which to choose label for the clusters
        - nearest = label of datapoint nearest to the centroid
        - most_common = most common label in the cluster

        :return `predictions`: list of predicted labels
        """

        assert mode in ['nearest', 'most_common'], f'unknown mode for assigning labels to clusters: {mode}'

        predictions = [0]*len(self.labels)
        
        if mode == 'nearest':
            for centroid, cluster in self.clusters.items():
                # take label of datapoint nearest to centroid
                datapoints = [(i, self.data[i]) for i in cluster]
                nearest_label_index, _ = min(datapoints, key=lambda datapoint: euclidean_distance(centroid, datapoint[1]))
                nearest_label = self.labels[nearest_label_index]
                for index in cluster:
                    predictions[index] = nearest_label

        elif mode == 'most_common':
            for centroid, cluster in self.clusters.items():
                # take most common label for datapoints in the cluster
                cluster_labels = [self.labels[i] for i in cluster]
                cluster_labels_counter = Counter(cluster_labels)
                most_common_label = cluster_labels_counter.most_common(1)[0][0]
                for index in cluster:
                    predictions[index] = most_common_label

        return predictions


    def fit_predict(self, data: list[tuple], labels: list[int]) -> list[int]:
        """
        Fits on data and assigns labels to clusters
        
        :param `data`: list of datapoints

        :param `labels`: list of labels

        :return `predictions`: list of predicted labels
        """

        assert isinstance(data, list), 'data should be a list'
        assert isinstance(data[0], tuple), 'datapoints should be tuples'
        self.data = data
        self.labels = labels

        # fit
        self.fit(10)

        # predict
        predictions = self.predict()

        return predictions
    