import abc
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import numpy.typing as npt
from collections import Counter
import pandas as pd

"""
Implement a classifier with required functions:

get_features: feature vector for each sample (1-hot, n-hot encodings or etc.)
fit(train_features, train_labels): to train the classifier
predict(test_features): to predict test labels 
"""


class CustomClassifier(abc.ABC):
    def __init__(self):
        self.counter = None

    def get_features(self, text_list: list[list[str]], n=1):
        """
        :param `text_list`: list of preprocessed, tokenized tweets
        :param `n`: length of gram in N-Hot encoded array, default 1
        :return `features_array`: 2D, N-Hot encoded numpy array of features per tweet
        :return `vocab`: array of unique ngrams in all of text_list
        """
        assert n >= 1, f'{self.get_features.__qualname__}: n should be 1 or larger'
        
        # generate vocabulary and convert tweets to ngrams
        vocab = set()
        ngram_tweet_list = []

        for tweet in text_list:
            ngram_tweet = []
            length = len(tweet)
            i = 0
            
            while i < length - n: # make sure we don't run out of space

                ngram = tuple(tweet[i:i+n])
                ngram_tweet.append(ngram)
                vocab.add(ngram) # tuples are ordered and unmutable
                
                i += 1
            
            ngram_tweet_list.append(ngram_tweet)

        vocab = tuple(vocab) # tuples are ordered and unmutable
        features_array = np.zeros(shape=(len(text_list), len(vocab))) # make a 2D matrix filled with zeros

        # loop over vocab and tweets in order of `features_array` dimensions
        for vocab_index, vocab_ngram in enumerate(vocab):
            for ngram_tweet_index, ngram_tweet in enumerate(ngram_tweet_list):
                for tweet_ngram in ngram_tweet:
                    if tweet_ngram == vocab_ngram:
                        features_array[ngram_tweet_index][vocab_index] += 1

        return features_array, vocab


    def tf_idf(self, text_feats):
        tfidf_transformer = TfidfTransformer().fit(text_feats)
        return tfidf_transformer.transform(text_feats)


    # @abc.abstractmethod
    # def fit(self, train_features, train_labels):
    #     pass

    # @abc.abstractmethod
    # def predict(self, test_features):
    #     pass
