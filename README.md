# twitter-sentiment-analysis
Sentiment triclass classifier for twitter, with predictions of positive, negative, or neutral.

### Overview

Training data: 45,101 tweets (tab-separated as tweetID--sentiment--text)

Test data: 3,531 tweets (same format)

Best classifier: Logistic regression with word2vec embeddings and TF-IDF unigrams as features.

Corresponding result: macro-averaged F1-score of 0.649.


# Important points for marker

I use word2vec Twitter embeddings, pretrained on 400m tweets, which can be found here: https://www.fredericgodin.com/software/

This is in python 2.

### Usage instructions

1. In this repo, download the pre-trained word embeddings using the link above.

2. Run `create_embeddings.py`. This will store the embeddings, tweet IDs, and tweet sentiment data in the `pickled_data` directory.

3. Run `classification.py`. This will run the best classifier for this dataset. You can modify the classifier quite easily in this file.

# Files in repo

Here are some descriptions of the other files and folders in the submission:

* ```feature_generation.py``` -- this includes the `Features` class which I use to construct the features to train the classifiers on, and the features to test the models on. A Features instance is created, and features are added by doing something like Features.AddBigrams, or Features.AddLexicons(...), or Features.AddEmbeddings(...). This can be seen in action in the `create_features` method in ```classification.py```.

* ```preprocess.py``` -- this includes all of the preprocessing methods. This is used in the original report, but is no longer in the code.

* ```create_embeddings.py``` -- this is used to create the feature vectors on my data from the loaded word embeddings, with the results being pickled as `pickled_data/data_embeddings.pkl`, which is a dictionary with keys as filenames (i.e. training or testing data), and values as the feature vectors. The feature generation makes use of these pickled embeddings to add word embeddings as features in the classifier.

* ```evaluation.py``` -- computes the macro-averaged F1-score for given test set results.

* ```pickled_data``` directory -- prior to running `create_embeddings.py`, this includes:
  *  `top_neg_words.pkl` and `top_pos_words.pkl`: the pre-identified top words most indicative of a positive or negative tweet. These are used in the feature generation class to add them as features.

## ```classification.py```

Here I provide a quick run through the this python code:

1. Load the pickled `top_pos_words` and `top_neg_words` that can be used if lexicons are a desired feature.

2. Read the training and test data.

3. Define `features`, which is a list of lists of features (each sublist is a list of the features desired in a given classifier).

4. Define `classifiers`, which is a list of classifiers as strings. These can be "maximum entropy" (logistic regression), "svm" (support vector machine), "nb" (naive Bayes), or "rf" (random forests). 

5. Loop through the classifiers. For each classifier, loop through the feature sets, and fit and predict for each set, outputting the result.
