import numpy as np 
import matplotlib.pyplot as plt
from nltk import bigrams
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.metrics import edit_distance
import spacy 
import scattertext as st
import pandas as pd
import pickle

def get_maxlen(sentences):
	
	max_len = 0
	for sent in sentences:
		if len(sent) > max_len:
			max_len = len(sent)
	return max_len

def import_glove(GLOVE_DIR, EMBEDDING_DIM):

	embeddings_index = {}
	with open(GLOVE_DIR + '/glove.twitter.27B.' + str(EMBEDDING_DIM) + 'd.txt') as f:
		for line in f:
		    values = line.split()
		    word = values[0]
		    coefs = np.asarray(values[1:], dtype='float32')
		    embeddings_index[word] = coefs

	return embeddings_index

def create_embedding(word2index, EMBEDDING_DIM, embeddings):
	embedding_matrix = np.zeros((len(word2index) + 1, EMBEDDING_DIM))
	for word, i in word2index.items():
		embedding_vector = embeddings.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector
	return embedding_matrix	     

def create_word2index(tweets):
	'''
		Input: list of tokenized tweets 
		Output: word2index dict
	'''
	all_words = []
	for tweet in tweets:
		all_words.extend(tweet)
	vocab = list(set(all_words))
	word2index = { word : i for i, word in enumerate(vocab) }
	word2index['NA'] = len(word2index) + 1
	return word2index

def create_bigram2index(tweets):
	'''
		Input: list of tokenized tweets 
		Output: bigram2index dict
	'''
	all_bigrams = []
	for tweet in tweets:
		tweet = list(bigrams(tweet))
		all_bigrams.extend(tweet)
	bigram_vocab = list(set(all_bigrams))
	bigram2index = { bigram : i for i, bigram  in enumerate(bigram_vocab) }
	bigram2index['NA'] = len(bigram2index) + 1
	return bigram2index

def padding(arr, maxlen, padding_pos, value):
	for l in range(len(arr)):
		if padding_pos == 'post':
			arr[l] += [value]*(maxlen - len(arr[l]))
		elif padding_pos == 'pre':
			arr[l] = [value]*(maxlen - len(arr[l])) + arr[l]
	return arr

def get_common_words(X_train, y_train, top_n):

	for i in range(len(X_train)):
		X_train[i] = unicode(X_train[i])
		if y_train[i] == 0:
			y_train[i] = 'neutral'
		elif y_train[i] == 1:
			y_train[i] = 'positive'
		else:
			y_train[i] = 'negative'

	twitter_df = pd.DataFrame({ 'tweet' : X_train, 'sentiment' : y_train })

	nlp = spacy.load('en')
	corpus = st.CorpusFromPandas(twitter_df,
							 	 category_col='sentiment',
							 	 text_col='tweet',
							 	 nlp=nlp).build()	

	term_freq_df = corpus.get_term_freq_df()
	term_freq_df['Positive score'] = corpus.get_scaled_f_scores('positive')
	term_freq_df['Negative score'] = corpus.get_scaled_f_scores('negative')

	top_pos_words = list(term_freq_df.sort_values(by='Positive score', ascending=False).index[:top_n])
	top_neg_words = list(term_freq_df.sort_values(by='Negative score', ascending=False).index[:top_n])

	return top_pos_words, top_neg_words

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

def lexicons_from_top_words(X, y, top_pos_words, top_neg_words):

	pos_word_count = np.zeros((len(X), len(top_pos_words)))
	neg_word_count = np.zeros((len(X), len(top_neg_words)))

	for i in range(len(X)):
		tweet = word_tokenize(X[i])
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
	
	return lexicons





