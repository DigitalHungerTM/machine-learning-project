from sklearn import svm
from abs_custom_classifier_with_feature_generator import CustomClassifier

"""
Implement a classifier with required functions:

fit(train_features, train_labels): to train the classifier
predict(test_features): to predict test labels 
"""


class SVMClassifier(CustomClassifier):
    def __init__(self, kernel='linear', **kwargs):
        """ """
        super().__init__()

        self.classifier = svm.SVC(kernel=kernel, **kwargs)

    def fit(self, train_features, train_labels):
        self.classifier.fit(train_features, train_labels)

    def predict(self, test_features):
        return self.classifier.predict(test_features)
