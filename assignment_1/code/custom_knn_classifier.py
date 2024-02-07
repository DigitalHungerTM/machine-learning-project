import numpy as np
import scipy
from abs_custom_classifier_with_feature_generator import CustomClassifier

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
        """ Predict classes with provided test features """

        assert self.is_trained, 'Model must be trained before predicting'

        predictions = []

        # 2D array of distances between all test and all training samples
        # Shape (Test X Train)
        distance_values = scipy.spatial.distance.cdist()

        # Use provided function by replacing X and Y, to calculate distance between test feature(s) and train feature(s)
        # You can use the function either by giving two matrices (All test features, All train features)
        # or by passing a matrix and a vector: (A test feature, All train features)
        distance_values = scipy.spatial.distance.cdist()

        predictions = []
        return predictions
