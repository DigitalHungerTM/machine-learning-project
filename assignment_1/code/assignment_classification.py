import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn import metrics
from custom_knn_classifier import CustomKNN
from custom_naive_bayes import CustomNaiveBayes
from sklearn_svm_classifier import SVMClassifier
from sklearn.utils import shuffle
from unidecode import unidecode


##################################################################
####################### DATASET FUNCTIONS ########################
##################################################################

def read_dataset(folder: str, split: str):
    """
    reads tsv file and gives tweets and their labels
    
    :param `folder`: relative folder path
    :param `split`: filename without extension
    :return `texts`: list of tweets
    :return `labels`: corresponding labels
    """
    inputf = open(os.path.join(folder, f'{split}.tsv'), encoding='utf-8')
    inputdf = pd.read_csv(inputf, sep="\t", encoding='utf-8', header=0)

    texts = inputdf.tweet_text.to_list()
    labels = inputdf.class_label.to_list()

    assert len(texts) == len(labels), 'Text and label files should have same number of lines..'
    # print(f'Number of samples: {len(texts)}')

    return texts, labels

def preprocess_dataset(text_list: list[str]):
    """
    Return the list of sentences after preprocessing. Example:
    >>> preprocess_dataset(['The quick Brown fOx #HASTAG-1234 @USER-XYZ'])
    ['the quick brown fox']
    makes everything lowercase and removes:
    - user tags
    - hashtags
    - links
    - punctuation
    and then tokenizes
    """

    preprocessed_text_list = []
    for tweet in text_list:
        tweet = " ".join([word for word in tweet.split()
                            if not word.startswith("@") # remove tags
                            and not word.startswith("#") # remove hashtags
                            and not word.startswith("http") # remove links
                        ]).lower() # de-capitalize
        tweet = re.sub(r'[^\w\s]', '', tweet) # remove anything that is not a word or whitespace (punctuation and emojis)
        tweet = unidecode(tweet) # replace accented letters (e.g., ë and à) with their unicode counterparts (e and a)
        # tweet = tweet.split() # tokenize, remove for character based (also remove the vocab pickle or this won't work)
        preprocessed_text_list.append(tweet)

    return preprocessed_text_list

##################################################################
####################### EVALUATION METRICS #######################
##################################################################


def evaluate(true_labels=[1,0,3,2,0], predicted_labels=[1,3,2,2,0]):
    """
    Print accuracy, precision, recall and f1 metrics for each classes and macro average.

    :param `true_labels`: true labels from the test set
    :param `predicted_labels`: predicted labels from the test set
    :return `csv_string`: csv formatted string of accuracy and macro f1 score
    """
    
    confusion_matrix = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels)

    # accuracy
    accuracy = confusion_matrix.diagonal().sum() / confusion_matrix.sum()

    # precision
    precision = confusion_matrix.diagonal() / confusion_matrix.sum(axis = 0)
    macro_precision = np.mean(precision)

    # recall
    recall = confusion_matrix.diagonal() / confusion_matrix.sum(axis = 1)
    macro_recall = np.mean(recall)

    # f1 score
    f1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    macro_f1 = np.mean(f1)
    
    # do some nice string formatting
    print(
        f"""
accuracy:  {np.round(accuracy, 3)}
precision: {np.round(list(precision), 3)}
recall:    {np.round(list(recall), 3)}
f1:        {np.round(list(f1), 3)}

macro avg:
precision: {np.round(macro_precision, 3)}
recall:    {np.round(macro_recall, 3)}
f1:        {np.round(macro_f1, 3)}
        """, file=sys.stderr
    )

    csv_string = f"{accuracy},{macro_f1}"
    return csv_string


def train_test(classifier='svm', n=1, k=5, distance_metric='cosine'):
    """
    loads data, preprocesses, fits on train data and predicts labels for test data,
    then evaluates
    :param `classifier`: type of classifier you want to use, default is svm
    :param `n`: number of tokens that the n-grams should contain, default is 1
    :param `k`: number of nearest neighbours for the knn classifier, default is 5
    """
    print("reading data", file=sys.stderr)
    train_data, train_labels = read_dataset('data', 'CT22_dutch_1B_claim_train')
    test_data, test_labels = read_dataset('data', 'CT22_dutch_1B_claim_dev_test')

    print("processing data", file=sys.stderr)
    train_data = preprocess_dataset(train_data)
    test_data = preprocess_dataset(test_data)


    # Create a your custom classifier
    if classifier == 'svm':
        cls = SVMClassifier(kernel='linear')
    elif classifier == 'naive_bayes':
        cls = CustomNaiveBayes()
    elif classifier == 'knn':
        cls = CustomKNN(k, distance_metric)

    print("getting features", file=sys.stderr)
    train_feats = cls.get_features(train_data, n)
    test_feats = cls.get_features(test_data, n)

    
    print("training classifier", file=sys.stderr)
    cls.fit(train_feats, train_labels)

    print("predicting labels", file=sys.stderr)
    predicted_test_labels = cls.predict(test_feats)

    print("evaluating")
    print(f"{n},{k},{evaluate(test_labels, predicted_test_labels)}")


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
    # settings
    n=4
    k=15
    distance_metric = 'euclidean'


    print(f"type='knn', {n=}, {k=}", file=sys.stderr)
    train_test('knn', n, k, distance_metric=distance_metric)


if __name__ == "__main__":
    main()
