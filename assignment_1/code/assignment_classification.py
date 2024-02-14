import os
import re
from typing import Literal
import numpy as np
import pandas as pd
from sklearn import metrics
from custom_knn_classifier import CustomKNN
from sklearn_svm_classifier import SVMClassifier
from sklearn.utils import shuffle
from unidecode import unidecode
from math import floor


def read_dataset(folder: str, split: str):
    """
    reads tsv file and gives documents and their labels
    
    :param `folder`: relative folder path
    :param `split`: filename without extension
    :return `documents`: list of documents
    :return `labels`: corresponding labels
    """
    inputf = open(os.path.join(folder, f'{split}.tsv'), encoding='utf-8')
    inputdf = pd.read_csv(inputf, sep="\t", encoding='utf-8', header=0)

    documents = inputdf.tweet_text.to_list()
    labels = inputdf.class_label.to_list()

    assert len(documents) == len(labels), 'Text and label files should have same number of lines..'
    # print(f'Number of samples: {len(texts)}')

    return documents, labels


def preprocess_dataset(documents: list[str]):
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
    for doc in documents:
        doc = " ".join([word for word in doc.split()
                            if not word.startswith("@") # remove tags
                            and not word.startswith("#") # remove hashtags
                            and not word.startswith("http") # remove links
                        ]).lower() # de-capitalize
        doc = re.sub(r'[^\w\s]', '', doc) # remove anything that is not a word or whitespace (punctuation and emojis)
        doc = unidecode(doc) # replace accented letters (e.g., ë and à) with their unicode counterparts (e and a)
        # doc = doc.split() # tokenize, remove for character based (also remove the vocab pickle or this won't work)
        preprocessed_text_list.append(doc)

    return preprocessed_text_list


def evaluate(true_labels, predicted_labels) -> dict[Literal['accuracy', 'precision', 'macro_precision', 'recall', 'macro_recall', 'f1', 'macro_f1'], float]:
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
               classifier: Literal['svm', 'knn']='svm',
               n: int=2,
               k: int=5,
               distance_metric: Literal['euclidean', 'cosine']='euclidean',
               ):
    """
    loads data, preprocesses, fits on train data and predicts labels for test data,
    then evaluates
    :param `classifier`: type of classifier you want to use, default is svm
    :param `n`: number of tokens that the n-grams should contain, default is 1
    :param `k`: number of nearest neighbours for the knn classifier, default is 5
    """
    # do some input checking
    assert classifier in ['svm', 'knn'], f'{classifier} is not a known classifier'
    assert n >= 1, 'n is not 1 or larger'
    assert k >= 1, 'k is not 1 or larger'
    if classifier == 'knn':
        assert distance_metric in ['euclidean', 'cosine'], f'{distance_metric} is not a valid distance metric'

    print("processing data")
    preproc_train_data = preprocess_dataset(data['train']['documents'])
    preproc_test_data = preprocess_dataset(data['test']['documents'])

    if classifier == 'svm':
        cls = SVMClassifier(kernel='linear')
    elif classifier == 'knn':
        cls = CustomKNN(k, distance_metric)

    print("getting features")
    train_feats = cls.get_features(preproc_train_data, n)
    test_feats = cls.get_features(preproc_test_data, n)

    print("training classifier")
    cls.fit(train_feats, data['train']['labels'])

    print("predicting labels")
    predicted_test_labels = cls.predict(test_feats)

    print("evaluating")
    results = evaluate(data['test']['labels'], predicted_test_labels)
    return results


def cross_validate(data, classifier, n, k, distance_metric, n_fold=10):
    """
    cross validates n folds of the classifier
    :param `data`: dict containing train and test data and labels
    :param `n_fold`: number of folds for which the classifier should be compared, default 10
    :param `classifier`: type of classifier
    """

    # Shuffle train data and train labels with the same indexes (random_state for reproducing same shuffling)
    data['train']['documents'], data['train']['labels'] = shuffle(data['train']['documents'], data['train']['labels'], random_state=0)

    # Split training data and labels into N folds
    scores = []
    start = 0
    end = len(data['train']['documents'])
    step = floor(len(data['train']['documents'])/n_fold)
    # if n-fold division result is not integer, last item is left out because of flooring
    # this shouldn't matter because the shuffle is random (except it isn't because `random_state=0`)

    for i in range(start, end, step):
        # make a new dict with the cuts
        fold_data = {
            'train': {
                'documents': data['train']['documents'][i:i+step],
                'labels': data['train']['labels'][i:i+step],
            },
            'test': {
                'documents': data['test']['documents'],
                'labels': data['test']['labels']
            }
        }
        results = train_test(fold_data, classifier, n, k, distance_metric)
        scores.append(results['macro_f1'])

    print(f'Average macro f1 score for {n_fold}-fold: {np.mean(np.array(scores))}')

    return np.mean(np.array(scores))


def main():
    train_data, train_labels = read_dataset('data', 'CT22_dutch_1B_claim_train')
    test_data, test_labels = read_dataset('data', 'CT22_dutch_1B_claim_dev_test')

    data_dict = {
        'train': {
            'documents': train_data,
            'labels': train_labels
        },
        'test': {
            'documents': test_data,
            'labels': test_labels
        }
    }
    
    # general
    classifier = 'knn'
    n = 2

    # knn
    k = 5
    distance_metric = 'cosine'

    # cross validation
    n_fold = 10

    # run train_test first to make sure the vocab is generated
    train_test(    data_dict, classifier, n, k, distance_metric)
    cross_validate(data_dict, classifier, n, k, distance_metric, n_fold)


if __name__ == "__main__":
    main()
