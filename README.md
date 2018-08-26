# twitter-sentiment-analysis
Sentiment classifier for twitter, with predictions of positive, negative, or neutral.

### Overview

Training data: 45,101 tweets (tab-separated as tweetID \t sentiment \t text)

Test data: 3,531 tweets (same format)

Best classifier: Logistic regression with word2vec embeddings and TF-IDF unigrams as features.

Corresponding result: macro-averaged F1-score of 0.649.

Note that the report in the pdf document was written before this result was observed.

# Important points for marker

I use word2vec Twitter embeddings, pretrained on 400m tweets, which can be found here: https://www.fredericgodin.com/software/. The code written handles the word2vec embeddings trained on the GoogleNews dataset.

### Usage instructions

1. In this repo, download the pre-trained word embeddings using the link above.

2. Run `classification.py`. This will run the best classifier for this dataset. Running this code will run every classifier in the list on line 50. For feature choices, these have to entered as boolean arguments in `feature_pipeline()`. 

The default feature arguments in `feature_pipeline()` are:

1. word2vec = False
2. lexicons = False 
3. unigrams = False
4. bigrams = False 
5. tfidf_unigrams = False 
6. tfidf_bigrams = False

# Files in repo

Here are some descriptions of the other files and folders in the submission:

* ```feature_generation.py``` -- this includes the sklearn TransformerMixin classes for adding features. It also includes the `feature_pipeline()` function, which takes as inputs the Twitter data and the desired features, and returns a transformed array of features for classification.

* ```preprocess.py``` -- this includes all of the preprocessing methods. 

* ```evaluation.py``` -- computes the macro-averaged F1-score for given test set results.
