#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import evaluation
import gensim
import re
import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from feature_generation import *

def create_features(X_train, y_train, X_test, y_test, features):
    train_feats = Features(X_train, y_train, X_test, y_test, is_train=True)
    test_feats = Features(X_train, y_train, X_test, y_test, is_train=False)
    if "embeddings" in features:
        train_feats.AddEmbeddings(data_type="train")
        test_feats.AddEmbeddings(data_type="test")
    if "lexicons" in features:
        train_feats.AddLexicons(1000, top_pos_words, top_neg_words)
        test_feats.AddLexicons(1000, top_pos_words, top_neg_words)
    if "unigrams" in features:
        train_feats.AddUnigrams()
        test_feats.AddUnigrams()
    if "bigrams" in features:
        train_feats.AddBigrams()
        test_feats.AddBigrams()
    if "tfidf-unigrams" in features:
        train_feats.AddTfidfUnigrams()
        test_feats.AddTfidfUnigrams()
    if "tfidf-bigrams" in features:
        train_feats.AddTfidfBigrams()    
        test_feats.AddTfidfBigrams()
    X_train = train_feats.features 
    X_test = test_feats.features   
    return X_train, X_test  

DIR = "data/"
def read_unprocessed_data(DIR, FILE_NAME):
    X = []
    y = []
    ID = []
    with open(DIR + FILE_NAME, 'r') as f:
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

with open("pickled_data/data_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
with open("pickled_data/tweetids.pkl", "rb") as f:
    tweetids = pickle.load(f)
with open("pickled_data/tweetgts.pkl", "rb") as f:
    tweetgts = pickle.load(f)
with open('pickled_data/top_pos_words.pkl', 'rb') as f:
    top_pos_words = pickle.load(f)
with open('pickled_data/top_neg_words.pkl', 'rb') as f:
    top_neg_words = pickle.load(f)

X_train, y_train, ID_train = read_unprocessed_data(DIR, "twitter-training-data.txt")
X_train = [re.sub(r'[^A-Za-z0-9 ]+', '', tweet) for tweet in X_train]
X_test, y_test, ID_test = read_unprocessed_data(DIR, "twitter-test.txt")
X_test = [re.sub(r'[^A-Za-z0-9 ]+', '', tweet) for tweet in X_test]

features = [["embeddings", "tfidf-unigrams"]]
classifiers = ["maximum entropy"]

for classifier in classifiers:
    print(' ')
    if classifier == 'svm':
        print('Training ' + classifier)
        clf = LinearSVC(class_weight='balanced')
    elif classifier == 'nb':
        print('Training ' + classifier)
        clf = GaussianNB()
    elif classifier == 'rf':
        print('Training ' + classifier)
        clf = RandomForestClassifier(class_weight='balanced')
    elif classifier == 'maximum entropy':
        print('Training ' + classifier)
        clf = LogisticRegression(class_weight='balanced')
        
    for feats in features:
        X_train_tmp, X_test_tmp = create_features(X_train, y_train,
                                          X_test, y_test,
                                          features=feats)
        clf.fit(X_train_tmp, y_train)
        preds = clf.predict(X_test_tmp)

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

