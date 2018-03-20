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

		if self.features == None:

			vec = CountVectorizer(ngram_range=(1, 1))
			X_unigrams = self.X
			X_test_unigrams = self.X_test
			vec.fit(X_unigrams)
			unigrams = vec.transform(X_unigrams)
			if self.is_train:
				self.features = unigrams
			else:
				test_unigrams = vec.transform(X_test_unigrams)
				self.features = test_unigrams

		elif self.features != None:

			vec = CountVectorizer(ngram_range=(1, 1))
			X_unigrams = self.X
			X_test_unigrams = self.X_test
			vec.fit(X_unigrams)
			unigrams = vec.transform(X_unigrams)
			if self.is_train:
				self.features = sparse.hstack((self.features, unigrams))
			else:
				test_unigrams = vec.transform(X_test_unigrams)
				self.features = sparse.hstack((self.features, test_unigrams))

	def AddBigrams(self):

		if self.features == None:

			vec = CountVectorizer(ngram_range=(1, 2))
			X_bigrams = self.X
			X_test_bigrams = self.X_test
			vec.fit(X_bigrams)
			bigrams = vec.transform(X_bigrams)
			if self.is_train:				
				self.features = bigrams
			else:
				test_bigrams = vec.transform(X_test_bigrams)
				self.features = test_bigrams

		elif self.features != None:

			vec = CountVectorizer(ngram_range=(1, 2))
			X_bigrams = self.X
			X_test_bigrams = self.X_test
			vec.fit(X_bigrams)
			bigrams = vec.transform(X_bigrams)
			if self.is_train:
				self.features = sparse.hstack((self.features, bigrams))
			else:
				test_bigrams = vec.transform(X_test_bigrams)
				self.features = sparse.hstack((self.features, test_bigrams))

	def AddTfidfUnigrams(self):

		if self.features == None:

			vec = TfidfVectorizer(ngram_range=(1, 1))
			X_unigrams = self.X
			X_test_unigrams = self.X_test
			vec.fit(X_unigrams)
			unigrams = vec.transform(X_unigrams)
			if self.is_train:
				self.features = unigrams
			else:
				test_unigrams = vec.transform(X_test_unigrams)
				self.features = test_unigrams

		elif self.features != None:

			vec = CountVectorizer(ngram_range=(1, 1))
			X_unigrams = self.X
			X_test_unigrams = self.X_test
			vec.fit(X_unigrams)
			unigrams = vec.transform(X_unigrams)
			if self.is_train:
				self.features = sparse.hstack((self.features, unigrams))
			else:
				test_unigrams = vec.transform(X_test_unigrams)
				self.features = sparse.hstack((self.features, test_unigrams))

	def AddTfidfBigrams(self):

		if self.features == None:

			vec = TfidfVectorizer(ngram_range=(1, 2))
			X_bigrams = self.X
			X_test_bigrams = self.X_test
			vec.fit(X_bigrams)
			bigrams = vec.transform(X_bigrams)
			if self.is_train:				
				self.features = bigrams
			else:
				test_bigrams = vec.transform(X_test_bigrams)
				self.features = test_bigrams

		elif self.features != None:

			vec = CountVectorizer(ngram_range=(1, 2))
			X_bigrams = self.X
			X_test_bigrams = self.X_test
			vec.fit(X_bigrams)
			bigrams = vec.transform(X_bigrams)
			if self.is_train:
				self.features = sparse.hstack((self.features, bigrams))
			else:
				test_bigrams = vec.transform(X_test_bigrams)
				self.features = sparse.hstack((self.features, test_bigrams))


	def AddLexicons(self, top_n, top_pos_words, top_neg_words, pickled_path=None):

		lexicons = None
		
		if self.is_train == True:

			if pickled_path == None:

				pos_word_count = np.zeros((len(self.X), len(top_pos_words)))
				neg_word_count = np.zeros((len(self.X), len(top_neg_words)))

				for i in range(len(self.X)):
					tweet = word_tokenize(self.X[i])
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

				with open(pickled_path, 'r') as f:
					lexicons = pickle.load(f)

		else:

			if pickled_path == None:

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

				with open(pickled_path, 'r') as f:
					lexicons = pickle.load(f)

		if self.features == None:
			self.features = lexicons
		elif self.features != None:
			self.features = sparse.hstack((self.features, lexicons))


	def AddEmbeddings(self, vectors=None):

		word2vec = None
		
		if vectors != None:

			word2vec = vectors

		else:

			vocab = ' '.join(self.X_train)
			vocab = word_tokenize(vocab)
			vocab = list(set(vocab))
			word2index = { word : i for i, word in enumerate(vocab) }
			word2index['NA'] = len(word2index) + 1

			EMBEDDING_DIM = 200
			GLOVE_DIR = ''
			glove = import_glove(GLOVE_DIR, EMBEDDING_DIM) # From utils

			embedding = create_embedding(word2index=word2index, # From utils
							 EMBEDDING_DIM=EMBEDDING_DIM,
							 embeddings=glove)

			embedding = np.sum(embedding, axis=1) / EMBEDDING_DIM

			word2vector = { word : embedding[word2index[word]] for word in vocab }
			word2vector['NA'] = len(word2vector) + 1
			word2vex = word2vector

		train_vectors = [[word2vec[word] for word in word_tokenize(tweet)] for tweet in self.X]
		test_vectors = [[word2vec[word] if word in word2vec else word2vec['NA'] for word in word_tokenize(tweet)] for tweet in self.X_test]	
		max_len = 0
		for i in train_vectors:
			if len(i) > max_len:
				max_len = len(i)
		train_vectors = padding(train_vectors, maxlen=max_len, padding_pos='post', value=0)
		test_vectors = padding(test_vectors, maxlen=max_len, padding_pos='post', value=0)

		if self.features == None:
			if self.is_train == True:
				train_vectors = sparse.csr_matrix(train_vectors)
				self.features = train_vectors
			else:
				test_vectors = sparse.csr_matrix(test_vectors)
				self.features = test_vectors
		else:
			if self.is_train == True:
				train_vectors = sparse.csr_matrix(train_vectors)
				self.features = sparse.hstack((self.features, train_vectors))
			else:
				test_vectors = sparse.csr_matrix(test_vectors)
				self.features = sparse.hstack((self.features, test_vectors))





