import numpy as np
import scipy
from scipy.spatial.distance import cdist
from abs_custom_classifier_with_feature_generator import CustomClassifier
from statistics import mode

"""
Implement a KNN classifier with required functions:

fit(train_features, train_labels): to train the classifier
predict(test_features): to predict test labels 
"""


class CustomKNN(CustomClassifier):
    def __init__(self, k=5, distance_metric='cosine'):
        """ """
        super().__init__()

        self.k = k
        self.train_feats = None
        self.train_labels = None
        self.is_trained = False
        self.distance_metric = distance_metric

    def fit(self, train_feats, train_labels):
        """ Fit training data for classifier """

        self.train_feats = train_feats
        self.train_labels = np.array(train_labels)

        self.is_trained = True
        return self

    def predict(self, test_feats):
        """
        Predict labels of `test_feats`
        
        :param `test_feats`: features of test set
        :param `method`: method with which to calculate distances, defaults to 'cosine'
        :return `predictions`: list of predicted labels
        """

        assert self.is_trained, 'Model must be trained before predicting'

        # 2D array of distances between all test and all training samples
        # Shape (Test X Train)
        distance_values = cdist(test_feats, self.train_feats, metric=self.distance_metric)

        predictions = []
        for distances_list in distance_values:
            # get indexes for k nearest neighbours
            # from https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
            # k_nearest_neighbours_indexes = np.argpartition(distances_list, -self.k)[-self.k:]
            
            # my own version
            sorted_nearest_distances = sorted(list(enumerate(distances_list)), key=lambda x: x[-1])[-self.k:]
            k_nearest_neighbours_indexes = list(map(lambda x: x[0], sorted_nearest_distances))
            
            # map indexes to labels from the train set
            k_nearest_neighbours_labels = list(map(lambda x: self.train_labels[x], k_nearest_neighbours_indexes))
            
            # take the most common label
            most_common_label = mode(k_nearest_neighbours_labels)
            predictions.append(most_common_label)

        return predictions
