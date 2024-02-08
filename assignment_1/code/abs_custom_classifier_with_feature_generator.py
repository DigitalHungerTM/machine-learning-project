import abc
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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

    def gen_vocab(self, text_list: list[list[str]], n=1):
        """
        generates a vocab for a list of tweets according to the ngram value
        
        :param `text_list`: list of tokenized tweets
        :param `n`: number of tokens in ngram, default is 1
        :return `vocab`: ngram vocabulary of the tweets list
        """

        # generate vocabulary and convert tweets to ngrams
        vocab = set()

        for tweet in text_list:
            length = len(tweet)
            i = 0
            
            while i < length - n: # make sure we don't run out of space

                ngram = tuple(tweet[i:i+n])
                vocab.add(ngram) # tuples are ordered and unmutable
                
                i += 1
            
        return tuple(vocab) # tuples are ordered and unmutable
    
    def n_gram_ify(self, text_list: list[list[str]], n):
        """
        'ngram-ifies' the tweets in the `text_list`
        
        :param `text_list`: list of tweets
        :param `n`: number of tokens in ngram
        :return `ngram_tweet_list`: list of ngram-ified tweets
        """

        # generate vocabulary and convert tweets to ngrams
        ngram_tweet_list = []

        for tweet in text_list:
            ngram_tweet = []
            length = len(tweet)
            i = 0
            
            while i < length - n: # make sure we don't run out of space

                ngram = tuple(tweet[i:i+n])
                ngram_tweet.append(ngram)
                
                i += 1
            
            ngram_tweet_list.append(ngram_tweet)
        
        return ngram_tweet_list

    def get_features(self, text_list: list[list[str]], n=1):
        """
        :param `text_list`: list of preprocessed, tokenized tweets
        :param `n`: length of gram in N-Hot encoded array, default 1
        :return `features_array`: 2D, N-Hot encoded numpy array of features per tweet
        """
        assert n >= 1, f'{self.get_features.__qualname__}: n should be 1 or larger'
        VOCAB_GENERATED = False
        print("getting features")

        # load vocab from pickle if it exists
        if path.exists("data/vocab.pickle") and path.isfile("data/vocab.pickle"):
            print("vocab pickle found, loading vocab pickle")
            with open("data/vocab.pickle", "rb") as pickle_file:
                vocab = pickle.load(pickle_file)

            # check if vocab has the same value as given n for ngrams
            if len(vocab[0]) != n:
                print("changed n, regenerating vocab")
                vocab = self.gen_vocab(text_list, n)
                VOCAB_GENERATED = True

        else:
            print("vocab pickle not found, generating vocab")
            vocab = self.gen_vocab(text_list, n)
            VOCAB_GENERATED = True

        # convert tweets to ngrams
        ngram_tweet_list = self.n_gram_ify(text_list, n)

        features_array = np.zeros(shape=(len(ngram_tweet_list), len(vocab))) # make a 2D matrix filled with zeros

        # loop over vocab and tweets in order of `features_array` dimensions
        for vocab_index, vocab_ngram in enumerate(vocab):
            for ngram_tweet_index, ngram_tweet in enumerate(ngram_tweet_list):
                for tweet_ngram in ngram_tweet:
                    if tweet_ngram == vocab_ngram:
                        features_array[ngram_tweet_index][vocab_index] += 1

        # save new pickle dump if a vocab was generated
        if VOCAB_GENERATED:
            with open("data/vocab.pickle", "wb") as pickle_write_file:
                pickle.dump(vocab, pickle_write_file)
        return features_array


    def tf_idf(self, text_feats):
        tfidf_transformer = TfidfTransformer().fit(text_feats)
        return tfidf_transformer.transform(text_feats)


    # @abc.abstractmethod
    # def fit(self, train_features, train_labels):
    #     pass

    # @abc.abstractmethod
    # def predict(self, test_features):
    #     pass
