import numpy as np
import gensim
import itertools

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

from nltk.tokenize import word_tokenize
import nltk


WORD2VEC_PATH = "data/GoogleNews-vectors-negative300.bin.gz"
TOP_N = 100


def join_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))


def feature_pipeline(X, X_train, y_train, word2vec=True, lexicons=True, unigrams=True, pos_tags=True,
                        bigrams=True, tfidf_unigrams=True, tfidf_bigrams=True):
    transformer = []
    if word2vec:
        transformer.append(("w2v", Word2VecEmbeddingsTransformer(WORD2VEC_PATH)))
    if lexicons:
        transformer.append(("lex", LexiconsTransformer(X_train, y_train, TOP_N)))
    if unigrams:
        transformer.append(("unicount", CountVectorizer(ngram_range=(1, 1))))
    if bigrams:
        transformer.append(("bicount", CountVectorizer(ngram_range=(1, 2))))
    if tfidf_unigrams:
        transformer.append(("tfidf_uni", TfidfVectorizer(ngram_range=(1, 1))))
    if tfidf_bigrams:
        transformer.append(("tfidf_bi", TfidfVectorizer(ngram_range=(1, 2))))
    if pos_tags:
        transformer.append(("pos", POSTagsTransformer()))

    pipeline = Pipeline([
        ("linguistic_union", FeatureUnion(
            transformer_list = transformer
        )),
    ])

    return pipeline


class Word2VecEmbeddingsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec_path):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True, unicode_errors="ignore")
        self.num_features = self.model.syn0.shape[1]
        self.index2word_set = set(self.model.index2word)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vecs = []

        for sent in X:
            feature_vec = np.zeros((self.num_features,), dtype="float32")
            nwords = 0

            for word in sent:
                if word != "" and word in self.index2word_set:
                    nwords += 1
                    feature_vec = np.add(feature_vec, self.model[word])

            if nwords > 1:
                feature_vec = np.divide(feature_vec, nwords)

            vecs.append(feature_vec)

        return vecs


class LexiconsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, X_train, y_train, top_n):
        # Use this to get the positive and negative lexicons that
        # will be applied to the test set
        self.top_n = top_n
        X_train = [word_tokenize(sent) for sent in X_train]
        #X_train = list(itertools.chain.from_iterable(X_train))
        print(len(X_train), len(y_train))
        X_train_positive = [X_train[i] for i in range(len(X_train)) if y_train[i] == 0]
        X_train_negative = [X_train[i] for i in range(len(X_train)) if y_train[i] == 1]

        X_train_positive = join_lists(X_train_positive)
        X_train_negative = join_lists(X_train_negative)

        pos_lexicons = nltk.FreqDist(X_train_positive).most_common(self.top_n)
        neg_lexicons = nltk.FreqDist(X_train_negative).most_common(self.top_n)

        # Create as a dictionary for efficient indexing when creating the
        # feature space in transform()
        self.pos_lexicons = {word[0]: (1, i) for i, word in enumerate(pos_lexicons)}
        self.neg_lexicons = {word[0]: (1, i) for i, word in enumerate(neg_lexicons)}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Feature space is a (len(X) x 2top_n) array
        # Element (i, j) of the array is the number of occurrences of lexicon
        # j in sentence i of the corpus
        def get_feature_space(sentiment):
            # Get the feature space for the input sentiment
            feature_space = np.zeros((len(X), self.top_n))
            for i, sent in enumerate(X):
                for word in word_tokenize(sent):
                    try:
                        if sentiment == "positive":
                            val = self.pos_lexicons[word]
                        elif sentiment == "negative":
                            val = self.neg_lexicons[word]
                        counter = val[0]  # val[0] is always 1
                        position = val[1] # val[1] is the position of the lexicon in the feature space
                        feature_space[i][position] += counter
                    except KeyError:
                        counter = 0
            return feature_space

        pos_feature_space = get_feature_space(sentiment="positive")
        neg_feature_space = get_feature_space(sentiment="negative")

        # Concatenate the positive and negative feature spaces to create a
        # single feature space for the lexicons
        feature_space = np.hstack((pos_feature_space, neg_feature_space))

        return feature_space


class POSTagsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dummy = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # The feature space is a (len(X) x n) array, where n is the total number
        # of POS tags in the corpus
        # Element (i,j) in array is the number of occurrences of POS tag j in sentence i
        X_pos = [nltk.pos_tag(sent) for sent in X]
        all_pos_tags = list(itertools.chain.from_iterable(X_pos))
        all_pos_tags = list(set([tag[1] for tag in all_pos_tags]))
        num_pos_tags = len(all_pos_tags)

        pos_tag_feats = { i: { tag: 0 for tag in all_pos_tags } for i in range(len(X)) }
        for idx in range(len(X_pos)):
            for tagged_word in X_pos[idx]:
                tag = tagged_word[1]
                pos_tag_feats[idx][tag] += 1

        pos_features = [[pos_tag_feats[i][tag] for tag in all_pos_tags] for i in range(len(X))]

        return np.asarray(pos_features)
