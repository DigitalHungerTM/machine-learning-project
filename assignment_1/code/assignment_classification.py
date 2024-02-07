import os
import re
import numpy as np
import pandas as pd
from sklearn import metrics
from custom_knn_classifier import CustomKNN
from custom_naive_bayes import CustomNaiveBayes
from sklearn_svm_classifier import SVMClassifier
from sklearn.utils import shuffle


##################################################################
####################### DATASET FUNCTIONS ########################
##################################################################

def read_dataset(folder, split):
    print('***** Reading the dataset *****')

    inputf = open(os.path.join(folder, f'{split}.tsv'), encoding='utf-8')
    inputdf = pd.read_csv(inputf, sep="\t", encoding='utf-8', header=0)

    texts = inputdf.tweet_text.to_list()
    labels = inputdf.class_label.to_list()

    assert len(texts) == len(labels), 'Text and label files should have same number of lines..'
    print(f'Number of samples: {len(texts)}')

    return texts, labels

def preprocess_dataset(text_list: list[str]):
    """
    Return the list of sentences after preprocessing. Example:
    >>> preprocess_dataset(['the quick brown fox #HASTAG-1234 @USER-XYZ'])
    ['the quick brown fox']
    removes:
    - user tags
    - hashtags
    - links
    - punctuation
    and tokenizes
    """
    preprocessed_text_list = []
    for tweet in text_list:
        tweet = " ".join([word for word in tweet.split()
                            if not word.startswith("@") # remove tags
                            and not word.startswith("#") # remove hashtags
                            and not word.startswith("http") # remove links
                        ])
        tweet = re.sub(r'[^\w\s]', '', tweet) # remove anything that is not a word or whitespace (punctuation and emojis)
        tweet = tweet.split() # tokenize
        preprocessed_text_list.append(tweet)

    return preprocessed_text_list

##################################################################
####################### EVALUATION METRICS #######################
##################################################################


def evaluate(true_labels, predicted_labels):
    """
    Print accuracy, precision, recall and f1 metrics for each classes and macro average.
    >>> evaluate(true_labels=[1,0,3,2,0], predicted_labels=[1,3,2,2,0])
    accuracy: 0.6
    precision: [1. , 1. , 0.5, 0. ]
    recall: [0.5, 1. , 1. , 0. ]
    f1: [0.66666667, 1. , 0.66666667, 0.]

    macro avg:
    precision: 0.625
    recall: 0.625
    f1: 0.583

    accuracy: how often is the correct class predicted (percentage)
    for every class: how often is this class predicted correctly
    """
    # get all unique classes
    classes = tuple(set(true_labels))
    
    confusion_matrix = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels)

    print('***** Evaluation *****')


def train_test(classifier='svm'):
    # Read train and test data and generate tweet list together with label list
    train_data, train_labels = read_dataset('CT22_dutch_1B_claim', 'train')
    test_data, test_labels = read_dataset('CT22_dutch_1B_claim', 'test')

    # Preprocess train and test data
    #train_data = preprocess_dataset(train_data)
    #test_data = preprocess_dataset(test_data)


    # Create a your custom classifier
    if classifier == 'svm':
        cls = SVMClassifier(kernel='linear')
#    elif classifier == 'naive_bayes':
#        cls = CustomNaiveBayes()
#    elif classifier == 'knn':
#        cls = CustomKNN(k=5, distance_metric='cosine')

    # Generate features from train and test data
    # features: word count features per sentences as a 2D numpy array
    train_feats = cls.get_features(train_data)
    test_feats = cls.get_features(test_data)

    
    # Train classifier
    cls.fit(train_feats, train_labels)

    # Predict labels for test data by using trained classifier and features of the test data
    predicted_test_labels = cls.predict(test_feats)

    # Evaluate the classifier by comparing predicted test labels and true test labels
    evaluate(test_labels, predicted_test_labels)


def cross_validate(n_fold=10, classifier='svm'):
    """
    Implement N-fold (n_fold) cross-validation by randomly splitting taining data/features into N-fold
    Store f1-mesure scores in a list for result of each fold and return this list
    Check main() for using required functions
    >>> cross_validate(n_fold=3, classifier='svm')
    [0.5, 0.4, 0.6]
    """

    train_data, train_labels = read_dataset('tweet_classification_dataset', 'train')

    # Shuffle train data and tran labels with the same indexes (random_state for reproducing same shuffling)
    train_data, train_labels = shuffle(train_data, train_labels, random_state=0)

    # Split training data and labels into N folds

    scores = []

    print(f'Average [evaluation measures] for {n_fold}-fold: {np.mean(np.array(scores))}')

    return np.mean(np.array(scores))


def main():
    # train_test('svm')
    texts, _ = read_dataset("data", "train_cut")
    for text in preprocess_dataset(texts):
        print(text)


if __name__ == "__main__":
    main()
