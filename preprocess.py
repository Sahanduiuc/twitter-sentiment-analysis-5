import numpy as np 
import pickle 
import re
import sys

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
tknzr = TweetTokenizer()

def remove_urls(tweets):
	'''
		Removes URLs from tweets
	'''
	return [re.sub(r'([^\s]+)\.([^\s]+)|http([^\s]+)', '', tweet) for tweet in tweets]

def remove_punctuation(tweets):
	'''
		Removes full-stops and commas from tweets
		This ensures that emoticons and emotional punctuation are retained
	'''
	return [re.sub(r'[^A-Za-z0-9 ]', '', tweet) for tweet in tweets]

def reduce_large_spaces(tweets):
	'''
		Reduces spaces from 2 or more to 1 if they exist
	'''
	return [re.sub(r'(  +)', ' ', tweet) for tweet in tweets]

def remove_mentions(tweets, remove=None, replace=None):
	'''
		Removes mentions from tweets, replaces with MENTION
	'''
	if replace == True:
		return [re.sub(r'(@[A-Za-z0-9_]+)', 'MENTION', tweet) for tweet in tweets] 
	elif remove == True:
		return [re.sub(r'(@[A-Za-z0-9_]+)', '', tweet) for tweet in tweets] 
	else:
		return [re.sub(r'(@[A-Za-z0-9_]+)', 'MENTION', tweet) for tweet in tweets] 

def remove_hash_symbol(tweets):
	'''
		Removes the hash symbol from hashtags
		Keeps the word that was hashed
	'''
	return [re.sub(r'(#)', '', tweet) for tweet in tweets]

def identify_elongations(tweets):
	'''
		Finds vowel elongations and reduces to 2 occurrences
	'''
	return [re.sub(r'(.)\1+', r'\1\1', tweet) for tweet in tweets]

def replace_emojis(tweets):
	'''
		Happy emojis: :), (:, :-), (-:, :D, ;) 
		Sad emojis: :(, ):, :-), (-:, ;(, ); 
	'''
	tweets = [re.sub(r'(:-?\))|(\(-?:)|(;-?\))|(\(-?;)|(:-?D)|(;-?D)', ' HAPPY_EMOJI ', tweet) for tweet in tweets]
	tweets = [re.sub(r'(:-?\()|(\)-?:)|(;-?\()|(\)-?;)', ' SAD_EMOJI ', tweet) for tweet in tweets]
	return tweets

def lower_case(tweets):
	'''
		Lower-cases all words in the tweet 
		Does not lower-case words that are all uppercase
	'''
	return [tweet.lower() for tweet in tweets]

def remove_numeric_words(tweets):
	'''
		Remove all numbers and words with numbers in them
	'''
	return [re.sub(r'([^\s]+)[0-9]([^\s]+)|[0-9]', '', tweet) for tweet in tweets]

def remove_numbers(tweets):
	return [re.sub(r'[0-9]', '', tweet) for tweet in tweets]

def preprocess_pipeline(tweets):
	'''
		Implement the desired preprocessing steps defined above
	'''
	tweets = lower_case(tweets)
	tweets = remove_urls(tweets)
	tweets = remove_mentions(tweets, remove=True)
	tweets = remove_hash_symbol(tweets)
	tweets = identify_elongations(tweets)
	tweets = replace_emojis(tweets)
	tweets = remove_punctuation(tweets)
	tweets = remove_numeric_words(tweets)
	tweets = remove_numbers(tweets)
	tweets = reduce_large_spaces(tweets)
	return tweets








