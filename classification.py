#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
from utils import *
from preprocess import *
from feature_generation import Features
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from scipy import sparse

# TODO: load training data
DATA_DIR = ''
X_train, y_train, ID_train = read_unprocessed_data(DATA_DIR, 'twitter-training-data.txt')
X_dev, y_dev, ID_dev = read_unprocessed_data(DATA_DIR, 'twitter-dev-data.txt')

X_train = preprocess_pipeline(X_train)
X_dev = preprocess_pipeline(X_dev)

with open('pickled_data/word2vec.pkl', 'r') as f:
    glove = pickle.load(f)
with open('pickled_data/train_lexicons.pkl', 'r') as f:
    train_lexicons = pickle.load(f)
with open('pickled_data/top_pos_words.pkl', 'r') as f:
    top_pos_words = pickle.load(f)
with open('pickled_data/top_neg_words.pkl', 'r') as f:
    top_neg_words = pickle.load(f)

for classifier in ['Bigrams + Lexicons Logistic Regression', 'Unigrams + Lexicons Naive Bayes', 'TF-IDF Bigrams SVM']: 
    if classifier == 'Bigrams + Lexicons Logistic Regression':
        print('Training ' + classifier + '...')
        train_feats = Features(X_train, y_train, X_dev, y_dev, is_train=True)
        train_feats.AddBigrams()
        train_feats.AddLexicons(top_n=1000, top_pos_words=top_pos_words, top_neg_words=top_neg_words)
        train_feats = train_feats.features
        model = LogisticRegression(C=5)
        model = model.fit(train_feats, y_train)
    elif classifier == 'Unigrams + Lexicons Naive Bayes':
        print('Training ' + classifier + '...')
        train_feats = Features(X_train, y_train, X_dev, y_dev, is_train=True)
        train_feats.AddUnigrams()
        train_feats.AddLexicons(top_n=1000, top_pos_words=top_pos_words, top_neg_words=top_neg_words)
        train_feats = train_feats.features
        model = MultinomialNB(fit_prior=True, alpha=0.4)
        model = model.fit(train_feats, y_train)
    elif classifier == 'TF-IDF Bigrams SVM':
        print('Training ' + classifier + '...')
        train_feats = Features(X_train, y_train, X_dev, y_dev, is_train=True)
        train_feats.AddTfidfBigrams()
        train_feats = train_feats.features
        model = LinearSVC(loss='squared_hinge', C=2)
        model = model.fit(train_feats, y_train)

    for testset in testsets.testsets:

        X_test, y_test, ID_test = read_unprocessed_data(DATA_DIR, testset)
        X_test = preprocess_pipeline(X_test)

        if classifier == 'Bigrams + Lexicons Logistic Regression':
            test_feats = Features(X_train, y_train, X_test, y_test, is_train=False)
            test_feats.AddBigrams()
            test_feats.AddLexicons(top_n=1000, top_pos_words=top_pos_words, top_neg_words=top_neg_words)
            test_feats = test_feats.features
            preds = model.predict(test_feats)
        elif classifier == 'Unigrams + Lexicons Naive Bayes':
            test_feats = Features(X_train, y_train, X_test, y_test, is_train=False)
            test_feats.AddUnigrams()
            test_feats.AddLexicons(top_n=1000, top_pos_words=top_pos_words, top_neg_words=top_neg_words)
            test_feats = test_feats.features
            preds = model.predict(test_feats)
        elif classifier == 'TF-IDF Bigrams SVM':
            test_feats = Features(X_train, y_train, X_test, y_test, is_train=False)
            test_feats.AddTfidfBigrams()
            test_feats = test_feats.features
            preds = model.predict(test_feats)

        predictions = { }
        for i in range(len(ID_test)):
            ID = ID_test[i]
            pred = preds[i]
            if pred == 0:
                predictions[ID] = 'neutral'
            elif pred == 1:
                predictions[ID] = 'positive'
            else:
                predictions[ID] = 'negative'
        
        evaluation.evaluate(predictions, testset, classifier)
        #evaluation.confusion(predictions, testset, classifier)
