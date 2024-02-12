import abc
import numpy as np
import pickle
import os.path as path

"""
Implement a classifier with required functions:

get_features: feature vector for each sample (1-hot, n-hot encodings or etc.)
fit(train_features, train_labels): to train the classifier
predict(test_features): to predict test labels 
"""


class CustomClassifier(abc.ABC):
    def __init__(self):
        self.counter = None
        self.vocab_generated = False
        self.train_features_generated = False
    
    def n_gram_ify(self, text_list: list[list[str]] | list[str], n):
        """
        'ngram-ifies' the tweets in the `text_list`
        
        :param `text_list`: list of tokenized tweets (remove tokenizing for character based ngram-ifying)
        :param `n`: number of tokens in ngram
        :return `ngram_tweet_list`: list of ngram-ified tweets
        """

        padding = '.'*(n-1)

        # convert tweets to ngrams
        ngram_tweet_list = []

        for tweet in text_list:
            # pad tweet with meaningless data
            if isinstance(tweet, str):
                tweet = padding + tweet + padding
            elif isinstance(tweet, list):
                tweet = list(padding) + tweet + list(padding)
            
            ngram_tweet = []
            length = len(tweet)
            
            i = 0
            while i < length - n: # make sure we don't run out of space

                ngram = tuple(tweet[i:i+n])
                ngram_tweet.append(ngram)
                
                i += 1
            
            ngram_tweet_list.append(ngram_tweet)
        
        return ngram_tweet_list
    
    
    def get_vocab(self, ngram_tweet_list, n):
        """
        tries to find a pickle that has the same value for n as the tweet's ngrams.
        if not found, generates a new vocab and saves it as a pickle
        :param `ngram_tweet_list`: list of `ngram-ified` tweets
        :return `vocab`: tuple of ngrams
        """

        if path.exists("data/vocab.pickle") and path.isfile("data/vocab.pickle"):
            print("vocab pickle found, loading vocab pickle")
            with open("data/vocab.pickle", "rb") as pickle_file:
                vocab = pickle.load(pickle_file)

            # check if vocab has the same value as given n for ngrams
            if len(vocab[0]) == n:
                return vocab # early escape

        print("vocab pickle not found, or n changed. generating vocab")
        vocab = set()
        for ngram_tweet in ngram_tweet_list:
            for ngram in ngram_tweet:
             vocab.add(ngram)
        vocab = tuple(vocab) # tuples are ordered and unmutable

        with open("data/vocab.pickle", "wb") as pickle_out_file:
            pickle.dump(vocab, pickle_out_file)
        
        self.vocab_generated = True
        return vocab
    

    def get_features(self, text_list, n=1):
        """
        :param `text_list`: list of preprocessed tweets, either tokenized or not
        :param `n`: length of gram in N-Hot encoded array, default 1
        :return `features_array`: 2D, N-Hot encoded numpy array of features per tweet
        """
        assert n >= 1, f'{self.get_features.__qualname__}: n should be 1 or larger'
        
        # convert tweets to ngrams
        ngram_tweet_list = self.n_gram_ify(text_list, n)

        # get the vocab
        vocab = self.get_vocab(ngram_tweet_list, n)

        print("number of features:", len(vocab))

        features_array = np.zeros(shape=(len(ngram_tweet_list), len(vocab))) # make a 2D matrix filled with zeros

        # loop over vocab and tweets in order of `features_array` dimensions
        print("generating features on vocab")
        for vocab_index, vocab_ngram in enumerate(vocab):
            for ngram_tweet_index, ngram_tweet in enumerate(ngram_tweet_list):
                for tweet_ngram in ngram_tweet:
                    if tweet_ngram == vocab_ngram:
                        features_array[ngram_tweet_index][vocab_index] += 1

        return features_array

    # @abc.abstractmethod
    # def fit(self, train_features, train_labels):
    #     pass

    # @abc.abstractmethod
    # def predict(self, test_features):
    #     pass
