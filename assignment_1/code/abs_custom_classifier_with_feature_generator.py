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
    
    def n_gram_ify(self, text_list: list[list[str]], n):
        """
        'ngram-ifies' the tweets in the `text_list`
        
        :param `text_list`: list of tokenized tweets (remove tokenizing for character based ngram-ifying)
        :param `n`: number of tokens in ngram
        :return `ngram_tweet_list`: list of ngram-ified tweets
        """

        # convert tweets to ngrams
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
    
    
    def gen_vocab(self, ngram_tweet_list):
        """
        generate a vocab based on 'ngram-ified' tweets
        :param `ngram_tweet_list`: list of `ngram-ified` tweets
        :return `vocab`: tuple of ngrams
        """
        vocab = set()
        for ngram_tweet in ngram_tweet_list:
            for ngram in ngram_tweet:
             vocab.add(ngram)
        return tuple(vocab) # tuples are ordered and unmutable
    

    def get_features(self, text_list, n=1):
        """
        :param `text_list`: list of preprocessed tweets, either tokenized or not
        :param `n`: length of gram in N-Hot encoded array, default 1
        :return `features_array`: 2D, N-Hot encoded numpy array of features per tweet
        """
        assert n >= 1, f'{self.get_features.__qualname__}: n should be 1 or larger'
        VOCAB_GENERATED = False
        print("***** Getting features *****\n")
        
        # convert tweets to ngrams
        ngram_tweet_list = self.n_gram_ify(text_list, n)

        # load vocab from pickle if it exists
        if path.exists("data/vocab.pickle") and path.isfile("data/vocab.pickle"):
            print("vocab pickle found, loading vocab pickle")
            with open("data/vocab.pickle", "rb") as pickle_file:
                vocab = pickle.load(pickle_file)

            # check if vocab has the same value as given n for ngrams
            if len(vocab[0]) != n:
                print("changed n, regenerating vocab")
                vocab = self.gen_vocab(ngram_tweet_list)
                VOCAB_GENERATED = True

        else:
            print("vocab pickle not found, generating vocab")
            vocab = self.gen_vocab(ngram_tweet_list)
            VOCAB_GENERATED = True

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
