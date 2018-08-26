#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import evaluation
import gensim
import re
import numpy as np
import pickle
import preprocess
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from feature_generation import *

DIR = "data/"

def read_unprocessed_data(path, filename):
    X = []
    y = []
    ID = []
    with open(path + filename, 'r') as f:
        for tweet in f:
            tweet = tweet.split()
            ID.append(tweet[0])
            if tweet[1] == 'neutral':
                y.append(0)
            elif tweet[1] == 'positive':
                y.append(1)
            else:
                y.append(2)
            tweet = tweet[2:]
            tweet = ' '.join(tweet)
            X.append(tweet)
    return X, y, ID

def get_data(path):
    X_train, y_train, ID_train = read_unprocessed_data(path, filename="twitter-training-data.txt")
    X_test, y_test, ID_test = read_unprocessed_data(path, filename="twitter-test.txt")

    X_train = preprocess.preprocess_pipeline(X_train)
    X_test = preprocess.preprocess_pipeline(X_test)

    X = X_train + X_test
    y = y_train + y_test

    return X_train, X_test, X, y_train, y_test, y, ID_train, ID_test

X_train, X_test, X, y_train, y_test, y, ID_train, ID_test = get_data(DIR)

classifiers = ["maxent"]

for classifier in classifiers:
    print(" ")
    if classifier == 'svm':
        clf = LinearSVC(class_weight='balanced')
    elif classifier == 'nb':
        clf = GaussianNB()
    elif classifier == 'rf':
        clf = RandomForestClassifier(class_weight='balanced')
    elif classifier == 'maxent':
        clf = LogisticRegression(class_weight='balanced')

    print("Generating feature space...")
    feats = feature_pipeline(X, X_train, y_train, tfidf_unigrams=True, tfidf_bigrams=True, unigrams=True, bigrams=True, pos_tags=True)
    train_feats = feats[:45101,:]
    test_feats = feats[45101:,:]

    print(train_feats.shape, test_feats.shape)

    print("Training" + classifier)
    clf.fit(train_feats, y_train)
    preds = clf.predict(test_feats)

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
    evaluation.evaluate(predictions, "data/twitter-test.txt", classifier)

