import numpy as np
from abs_custom_classifier_with_feature_generator import CustomClassifier

"""
Implement a Naive Bayes classifier with required functions:

fit(train_features, train_labels): to train the classifier
predict(test_features): to predict test labels 
"""


class CustomNaiveBayes(CustomClassifier):

    def __init__(self, alpha=1.0):
        """ """
        super().__init__()

        self.alpha = alpha
        self.prior = None
        self.classifier = None

    def fit(self, train_feats, train_labels):
        """ Calculate the priors, fit training data for Naive Bayes classifier """

        self.classifier = []

        self.is_trained = True
        return self

    def predict(self, test_feats):
        """ Predict classes with provided test features """

        assert self.is_trained, 'Model must be trained before predicting'

        # Use the scikit-learn predict function
        predictions = []
        return predictions

