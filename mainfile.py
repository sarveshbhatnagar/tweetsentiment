import flask
from flask import request

import pickle
import tweepy
import csv
import unicodedata
import pandas as pd
from textblob import TextBlob
import sklearn

tweet_classify = pickle.load(open('finalized_model.sav', 'rb'))


def get_tweets(topic, max_count=10):
    """
    Returns 1000 tweets about a topic as a pandas dataframe.
    change number by using max_count
    """
    api_key = "J1MhAOtPirsUiacFFmgkSiJrn"
    api_secret = "8ZPIL6tn1ycj4US19XnBdwEjj4SatHNl8eLw48MERDqh8INGYP"
    auth = tweepy.OAuthHandler(api_key, api_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    count = 0
    date = []
    tweets = []
    df = pd.DataFrame(columns=['date', 'tweet'])
    for tweet in tweepy.Cursor(api.search, q=topic, count=1000,
                               lang="en",
                               tweet_mode='extended'
                               ).items():

        if(not tweet.full_text.startswith("RT @")):
            tweets.append(str(unicodedata.normalize(
                'NFKD', tweet.full_text).encode('ascii', 'ignore')))
            date.append(tweet.created_at)
            count += 1

        if(count > max_count):
            break

    return pd.DataFrame(data={'date': date, 'tweet': tweets})


def get_sentiment(text, only=False):
    """
    Given a text, returns sentiment.
    only = True means only sentiment is returned and not subjectivity.
    """
    text_blob = TextBlob(text)
    if(only):
        sentiment = text_blob.sentiment
        return sentiment[0]
    return text_blob.sentiment


def to_lower(text):
    """
    Converts text to lowercase
    """
    return text.lower()


def new_get_sentiment(classifier, tweet):
    return classifier.predict([tweet])[0]


def labeltoval(label):
    t = {'NEGATIVE': -1, 'POSITIVE': 1}
    return t[label]


def new_sentiment_topic(classifier, topic):
    tweets = get_tweets(topic, max_count=100)
    tweets['tweet'] = tweets.tweet.apply(to_lower)
    val = 0
    count = 0
    tweets['sentiment'] = tweets.tweet.apply(get_sentiment, only=True)
    for tweet in tweets['tweet']:
        x = labeltoval(new_get_sentiment(classifier, tweet))
        val += x
        count += 1
    val = val/count
    print((topic, val))
    return (topic, val)


app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    topic = request.args['topic']
    topic = str(topic)
    # new_sentiment_topic(tweet_classify, topic)
    try:
        return str(new_sentiment_topic(tweet_classify, topic))
    except Exception as e:
        return str(e)
