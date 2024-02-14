import abc
import numpy as np
import pickle
import os.path as path
from sklearn.feature_extraction.text import TfidfTransformer
from itertools import chain

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
        self.vocab = None
        self.ngram_documents = None
    
    def n_gram_ify(self, documents: list[list[str]] | list[str], n):
        """
        'ngram-ifies' the documents in the `text_list`
        
        :param `text_list`: list of tokenized documents (remove tokenizing for character based ngram-ifying)
        :param `n`: number of tokens in ngram
        :return `ngram_documents`: list of ngram-ified documents
        """

        padding = '.'*(n-1)

        # convert document to ngrams
        ngram_documents = []

        for doc in documents:
            # pad document with meaningless data to include documents that have a smaller length than 
            if isinstance(doc, str):
                doc = padding + doc + padding
            elif isinstance(doc, list):
                doc = list(padding) + doc + list(padding)
            
            ngram_doc = []
            length = len(doc)
            
            i = 0
            while i < length - n: # make sure we don't run out of space

                ngram = tuple(doc[i:i+n])
                ngram_doc.append(ngram)
                
                i += 1
            
            ngram_documents.append(ngram_doc)
        
        return ngram_documents
    
    def get_vocab(self, ngram_documents, n):
        """
        tries to find a pickle that has the same value for n as the document's ngrams.
        if not found, generates a new vocab and saves it as a pickle
        :param `ngram_documents`: list of `ngram-ified` documents
        :return `vocab`: tuple of ngrams
        """

        if path.exists("data/vocab.pickle") and path.isfile("data/vocab.pickle"):
            with open("data/vocab.pickle", "rb") as pickle_file:
                vocab_with_meta_data = pickle.load(pickle_file)

            # check if vocab has the same value as given n for ngrams
            if vocab_with_meta_data['n'] == n:
                return vocab_with_meta_data['vocab'] # early escape

        ngrams = list(chain.from_iterable(ngram_documents))
        vocab = set(ngrams) # only unique ngrams
        vocab = tuple(vocab) # tuples are ordered and unmutable

        # store vocab with correct metadata
        vocab_with_meta_data = {
            'n': n,
            'vocab': vocab,
        }

        with open("data/vocab.pickle", "wb") as pickle_out_file:
            pickle.dump(vocab_with_meta_data, pickle_out_file)
        
        self.vocab_generated = True
        return vocab
    
    def tf_idf(self, text_feats):
        """
        transforms a sparse matrix of (n_samples, n_features) into a tf-idf normalized
        sparse matrix of (n_samples, n_features)
        :param `text_feats`: matrix of N-hot encoded samples
        :return: tf-idf normalized matrix of N-hot encoded samples
        """
        tfidf_transformer = TfidfTransformer().fit(text_feats)
        return tfidf_transformer.transform(text_feats).toarray()

    def get_features(self, documents, n=1):
        """
        Get N-hot encoded features of documents according to vocabulary

        :param `documents`: list of preprocessed documents, either tokenized or not
        :param `n`: length of gram in N-Hot encoded array, default 1
        :return `features_array`: 2D, N-Hot encoded numpy array of features per document
        """

        # convert documents to ngrams
        self.ngram_documents = self.n_gram_ify(documents, n)

        # get the vocab
        self.vocab = self.get_vocab(self.ngram_documents, n)

        features_array = np.zeros(shape=(len(self.ngram_documents), len(self.vocab))) # make a 2D matrix filled with zeros

        for ngram_document_index, ngram_doc in enumerate(self.ngram_documents):
            for vocab_index, vocab_ngram in enumerate(self.vocab):
                features_array[ngram_document_index][vocab_index] = ngram_doc.count(vocab_ngram)

        return self.tf_idf(features_array)

    @abc.abstractmethod
    def fit(self, train_features, train_labels):
        pass

    @abc.abstractmethod
    def predict(self, test_features):
        pass
