import numpy as np 
import sys 
import pickle 
from utils import * 
from nltk.tokenize import word_tokenize
import timeit

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from scipy import sparse
from nltk.corpus import stopwords

class Features:
    def __init__(self, X, y, X_test, y_test, is_train=True):
        self.X = X
        self.X_test = X_test 
        self.y = y
        self.y_test = y_test
        self.is_train = is_train
        self.features = None

    def AddUnigrams(self):
        vec = CountVectorizer(ngram_range=(1, 1))
        X_unigrams = self.X
        X_test_unigrams = self.X_test 
        vec.fit(X_unigrams)
        unigrams = vec.transform(X_unigrams)
        test_unigrams = vec.transform(X_test_unigrams)
        if self.features == None:
            if self.is_train:
                self.features = unigrams 
            else:
                self.features = test_unigrams
        else:
            if self.is_train:
                self.features = sparse.hstack((self.features, unigrams))
            else:
                self.features = sparse.hstack((self.features, test_unigrams))

    def AddBigrams(self):
        vec = CountVectorizer(ngram_range=(1, 2))
        X_bigrams = self.X 
        X_test_bigrams = self.X_test 
        vec.fit(X_bigrams) 
        bigrams = vec.transform(X_bigrams)
        test_bigrams = vec.transform(X_test_bigrams)
        if self.features == None:
            if self.is_train:				
                self.features = bigrams
            else:
                self.features = test_bigrams
        else:
            if self.is_train:
                self.features = sparse.hstack((self.features, bigrams))
            else:
                self.features = sparse.hstack((self.features, test_bigrams))

    def AddTfidfUnigrams(self):
        vec = TfidfVectorizer(ngram_range=(1, 1))
        X_unigrams = self.X
        X_test_unigrams = self.X_test
        vec.fit(X_unigrams)
        unigrams = vec.transform(X_unigrams)
        test_unigrams = vec.transform(X_test_unigrams)
        if self.features == None:
            if self.is_train:
                self.features = unigrams
            else:
                self.features = test_unigrams
        else:
            if self.is_train:
                self.features = sparse.hstack((self.features, unigrams))
            else:
                self.features = sparse.hstack((self.features, test_unigrams))

    def AddTfidfBigrams(self):
        vec = TfidfVectorizer(ngram_range=(1, 2))
        X_bigrams = self.X
        X_test_bigrams = self.X_test
        vec.fit(X_bigrams)
        bigrams = vec.transform(X_bigrams)
        test_bigrams = vec.transform(X_test_bigrams)
            if self.features == None:
                if self.is_train:				
                    self.features = bigrams
                else:
                    self.features = test_bigrams
            else:
                if self.is_train:
                    self.features = sparse.hstack((self.features, bigrams))
                else:
                    self.features = sparse.hstack((self.features, test_bigrams))

    def AddLexicons(self, top_n, top_pos_words, top_neg_words, pickled_path=None):
        lexicons = None
        if pickled_path == None:
            if self.is_train:
                pos_word_count = np.zeros((len(self.X), len(top_pos_words)))
                neg_word_count = np.zeros((len(self.X), len(top_neg_words)))
                for i in range(len(self.X)):
                    tweet = word_tokenize(self.X[i])
                    for word in tweet:
                        if word in top_pos_words:
                            idx = top_pos_words[word]
                            pos_word_count[i][idx] += 1
                        if word in top_neg_words:
                            idx = top_neg_words[word]
                            neg_word_count[i][idx] += 1	
            else:
                pos_word_count = np.zeros((len(self.X_test), len(top_pos_words)))
                neg_word_count = np.zeros((len(self.X_test), len(top_neg_words)))
                    for i in range(len(self.X_test)):
                        tweet = word_tokenize(self.X_test[i])
                        for word in tweet:
                            if word in top_pos_words:
                                idx = top_pos_words[word]
                                pos_word_count[i][idx] +=1
                            if word in top_neg_words:
                                idx = top_neg_words[word]
                                neg_word_count[i][idx] += 1
            pos_lexicons = np.asarray(pos_word_count)
            neg_lexicons = np.asarray(neg_word_count)
            lexicons = np.concatenate((pos_lexicons, neg_lexicons), axis=1)
            lexicons = sparse.csr_matrix(lexicons)
        else:
            with open(pickled_path, "r") as f:
                lexicons = pickle.load(f)
            if self.features == None:
                self.features = lexicons
            else:
                self.features = sparse.hstack((self.features, lexicons))

    def AddEmbeddings(self, data_type):
        with open("pickled_data/data_embeddings.pkl", "rb") as f:
            data = pickle.load(f)

        if data_type == "train":
            embeddings = data["twitter-training-data.txt"]
        elif data_type == "test":
            embeddings = data["twitter-test.txt"]

        if self.features == None:
            self.features = embeddings 
        else:
            embeddings = sparse.csr_matrix(embeddings)
            self.features = sparse.hstack((self.features, embeddings))
