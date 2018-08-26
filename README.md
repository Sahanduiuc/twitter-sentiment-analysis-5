# twitter-sentiment-analysis
Sentiment triclass classifier for twitter, with predictions of positive, negative, or neutral.

### Overview

Training data: 45,101 tweets (tab-separated as tweetID--sentiment--text)

Test data: 3,531 tweets (same format)

Best classifier: Logistic regression with word2vec embeddings and TF-IDF unigrams as features.

Corresponding result: macro-averaged F1-score of 0.649.


# Important points for marker

I use word2vec Twitter embeddings, pretrained on 400m tweets, which can be found here: https://www.fredericgodin.com/software/

### Usage instructions

1. In this repo, download the pre-trained word embeddings using the link above.

2. Run `classification.py`. This will run the best classifier for this dataset. Running this code will run every classifier in the list on line 50. For feature choices, these have to entered as boolean arguments in `feature_pipeline()`. The default feature arguments are:

 a. word2vec = False
 b. lexicons = False 
 c. unigrams = False
 d. bigrams = False 
 e. tfidf_unigrams = False 
 f. tfidf_bigrams = False

# Files in repo

Here are some descriptions of the other files and folders in the submission:

* ```feature_generation.py``` -- this includes the sklearn TransformerMixin classes for adding features. It also includes the `feature_pipeline()` function, which takes as inputs the Twitter data and the desired features, and returns a transformed array of features for classification.

* ```preprocess.py``` -- this includes all of the preprocessing methods. 

* ```evaluation.py``` -- computes the macro-averaged F1-score for given test set results.
