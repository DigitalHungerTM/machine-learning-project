import os
import re
from typing import Literal
import numpy as np
import pandas as pd
from sklearn import metrics
from custom_knn_classifier import CustomKNN
from custom_naive_bayes import CustomNaiveBayes
from sklearn_svm_classifier import SVMClassifier
from sklearn.utils import shuffle
from unidecode import unidecode
from math import floor


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
    print("\nconfusion matrix\n", confusion_matrix)

    results = {}

    # accuracy
    results['accuracy'] = confusion_matrix.diagonal().sum() / confusion_matrix.sum()

    # precision
    results['precision'] = confusion_matrix.diagonal() / confusion_matrix.sum(axis=0)
    results['macro_precision'] = np.mean(results['precision'])

    # recall
    results['recall'] = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
    results['macro_recall'] = np.mean(results['recall'])

    # f1 score
    results['f1'] = np.nan_to_num(2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall']))
    results['macro_f1'] = np.mean(results['f1'])
    
    # do some nice string formatting
    print(
        f"""
accuracy:  {np.round(results['accuracy'], 3)}
precision: {np.round(list(results['precision']), 3)}
recall:    {np.round(list(results['recall']), 3)}
f1:        {np.round(list(results['f1']), 3)}

macro avg:
precision: {np.round(results['macro_precision'], 3)}
recall:    {np.round(results['macro_recall'], 3)}
f1:        {np.round(results['macro_f1'], 3)}
        """
    )
    return results


def train_test(data,
               classifier: Literal['svm', 'knn', 'naive_bayes']='svm',
               n: int=2,
               k: int=5,
               distance_metric: Literal['euclidean', 'cosine']='euclidean',
               nb_mode: Literal['gaussian', 'categorical']='gaussian',
               nb_alpha: float=1.0
               ):
    """
    loads data, preprocesses, fits on train data and predicts labels for test data,
    then evaluates
    :param `classifier`: type of classifier you want to use, default is svm
    :param `n`: number of tokens that the n-grams should contain, default is 1
    :param `k`: number of nearest neighbours for the knn classifier, default is 5
    """

    print("processing data")
    train_data = preprocess_dataset(data['train']['data'])
    test_data = preprocess_dataset(data['test']['data'])


    # Create a your custom classifier
    if classifier == 'svm':
        cls = SVMClassifier(kernel='linear')
    elif classifier == 'naive_bayes':
        cls = CustomNaiveBayes(mode=nb_mode, alpha=nb_alpha)
    elif classifier == 'knn':
        cls = CustomKNN(k, distance_metric)

    print("getting features")
    train_feats = cls.get_features(train_data, n)
    test_feats = cls.get_features(test_data, n)

    
    print("training classifier")
    cls.fit(train_feats, data['train']['labels'])

    print("predicting labels")
    predicted_test_labels = cls.predict(test_feats)

    print("evaluating")
    results = evaluate(data['test']['labels'], predicted_test_labels)
    return results


def cross_validate(data, n_fold=10, classifier='svm', n=2, k=5, distance_metric='euclidean', nb_mode='gaussian', nb_alpha=1.0):
    """
    cross validates n folds of the classifier
    :param `data`: dict containing train and test data and labels
    :param `n_fold`: number of folds for which the classifier should be compared, default 10
    :param `classifier`: type of classifier
    """

    # Shuffle train data and train labels with the same indexes (random_state for reproducing same shuffling)
    data['train']['data'], data['train']['labels'] = shuffle(data['train']['data'], data['train']['labels'], random_state=0)

    # Split training data and labels into N folds
    scores = []
    start = 0
    end = len(data['train']['data'])
    step = floor(len(data['train']['data'])/n_fold) # last item is left out because of flooring
    for i in range(start, end, step):
        # make a new dict with the cuts
        fold_data = {
            'train': {
                'data': data['train']['data'][i:i+step],
                'labels': data['train']['labels'][i:i+step],
            },
            'test': {
                'data': data['test']['data'],
                'labels': data['test']['labels']
            }
        }
        results = train_test(fold_data, classifier, n, k, distance_metric, nb_mode, nb_alpha)
        scores.append(results['macro_f1'])


    print(f'Average macro f1 score for {n_fold}-fold: {np.mean(np.array(scores))}')

    return np.mean(np.array(scores))


def main():
    train_data, train_labels = read_dataset('data', 'CT22_dutch_1B_claim_train')
    test_data, test_labels = read_dataset('data', 'CT22_dutch_1B_claim_dev_test')

    data_dict = {
        'train': {
            'data': train_data,
            'labels': train_labels
        },
        'test': {
            'data': test_data,
            'labels': test_labels
        }
    }

    train_test(data=data_dict, classifier='naive_bayes', n=1, k=5, distance_metric='euclidean', nb_mode='gaussian', nb_alpha=1.0)
    cross_validate(data=data_dict, n_fold=20, classifier='naive_bayes', n=1, nb_mode='gaussian', nb_alpha=1.0)


if __name__ == "__main__":
    main()
