# twitter-sentiment-analysis
Sentiment classifier for twitter

# Important points for marker

I use GloVe Twitter word embeddings, which can be found here: https://nlp.stanford.edu/projects/glove/

I used the download: glove.twitter.27B.zip

Unfortunately I have to use python 2. I am having issues with python 3 on my computer.

# Files in repo

Here are some descriptions of the other files and folders in the submission:

* ```feature_generation.py``` -- this includes the Features class which I use to construct the features to train the classifiers on, and the features to test the models on. An Features instance is created, and features are added by doing something like Features().AddBigrams, or Features.AddLexicons(...).

* ```preprocess.py``` -- this includes all of the preprocessing methods. In the main classification.py file I pass the preprocess_pipeline() method to the data, which can be found in this file.

* ```utils.py``` -- this includes various methods that are used in feature_generation.py. These will most likely not be initiated when you run my code, as they are used before objects have been pickled. (I have provided pickled objects, as explained below.)

* ```pickled_data``` directory -- this includes a number of serialised objects that took 15 minutes or more to create. These are the positive and negative lexicons sparse matrix from the training set, the glove embeddings for the training set, and some other objects. I read these in at the beginning of the ```classification.py``` file.


## classification.py

Here I provide a quick run through the this python code:

1. Read in train and dev sets. (Assumes the data is in the current working directory.) Produce the labels and the IDs as well.
2. Preprocess the train and dev sets.
3. Read in the serialised objects.
4. Train and test classifiers!
