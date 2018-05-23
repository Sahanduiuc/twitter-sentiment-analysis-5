#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import evaluation
import gensim
import re
import numpy as np
import pickle

def loadW2vModel():
    global model
    global num_features
    global index2word_set

    model = gensim.models.KeyedVectors.load_word2vec_format('word2vec_twitter_model/word2vec_twitter_model.bin', binary=True, unicode_errors='ignore')
    num_features = model.syn0.shape[1]
    index2word_set = set(model.index2word)

def makeFeatureVec(words, getsum = 0): # Function to average all of the word vectors in a given sentence, set getsum to 1 if sum wanted instead
    global model
    global num_features
    global index2word_set

    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.

    for word in words:
        if word != '' and word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])

    if nwords > 1.0 and getsum == 0:
        featureVec = np.divide(featureVec,nwords)

    return featureVec

loadW2vModel()

data = {}
tweetids = {}
tweetgts = {}

for dataset in ['twitter-training-data.txt', 'twitter-test.txt']:
    data[dataset] = []
    tweetids[dataset] = []
    tweetgts[dataset] = []
    with open(dataset, 'r') as fh:
        for line in fh:
            linedata = line.strip().split('\t')
            tweetids[dataset].append(linedata[0])
            tweetgts[dataset].append(linedata[1])
            text = re.sub(r'[^A-Za-z0-9 ]+', '', linedata[2])
            data[dataset].append(makeFeatureVec(text.lower().split()))

with open("pickled_data/data_embeddings.pkl", "wb") as f:
    pickle.dump(data, f)
with open("pickled_data/tweetids.pkl", "wb") as f:
    pickle.dump(tweetids, f)
with open("pickled_data/tweetgts.pkl", "wb") as f:
    pickle.dump(tweetgts, f)


