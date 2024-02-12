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
        :return `predictions`: list of predicted labels
        """

        assert self.is_trained, 'Model must be trained before predicting'

        # 2D array of distances between all test and all training samples
        # Shape (Test X Train)
        distance_matrix = cdist(test_feats, self.train_feats, metric=self.distance_metric)
        print(np.shape(distance_matrix))

        predictions = []

        for distance_vector in distance_matrix:
            # get distances to all train vectors
            # (not sure if it works that way)

            # store indexes
            distance_vector_with_indexes = list(enumerate(distance_vector))

            # sort by distance
            sorted_distances_with_indexes = sorted(distance_vector_with_indexes, key=lambda x: x[-1])

            # extract indexes
            sorted_indexes = [i for i, _ in sorted_distances_with_indexes]

            # truncate indexes for k lowest distance (nearer means lower distance)
            k_nearest_indexes = sorted_indexes[:self.k]

            # map indexes to labels
            k_nearest_labels = [self.train_labels[i] for i in k_nearest_indexes]

            # take the most common label
            prediction = mode(k_nearest_labels)

            predictions.append(prediction)

        return predictions
