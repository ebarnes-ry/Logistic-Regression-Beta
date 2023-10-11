import nltk
from os import getcwd
#import w1_unittest

nltk.download('twitter_samples')
nltk.download('stopwords')

filepath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filepath)

import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from nltk.tokenize import TweetTokenizer

import re
import string

def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation): 
        # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)
    return tweets_clean

def build_freqs(tweets, ys):
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

all_pos = twitter_samples.strings('positive_tweets.json')
all_neg = twitter_samples.strings('negative_tweets.json')

print(type(twitter_samples))
print(type(all_pos))

test_pos = all_pos[4000:]
train_pos = all_pos[:4000]
test_neg = all_neg[4000:]
train_neg = all_neg[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)),axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)),axis=0)

#test split should be 20% train set should be 80%

print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))

freqs = build_freqs(train_x, train_y)

print("type freqs = " + str(type(freqs)))
print("length freqs = " + str(len(freqs.keys())))

print(train_x[0])

print(process_tweet(train_x[0]))

def sigmoid(z):
    h = 1/(1+np.exp(-z))
    return h

def gradient (x, y, theta, alpha, num_iters):
    m = len(y)
    for i in range(0, num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        J = (-1/m)*np.sum(y*np.log(h)+(1-y)*np.log(1-h))
        theta = theta-(alpha/m)*np.dot(x.T, (h-y))
    J = float(J)
    return J, theta

#check
np.random.seed(1)
#X input is 10 by 3 dim with 1s for biases
tmp_X = np.append(np.ones((10,1)), np.random.rand(10,2)*2000, axis=1)
#Y labels are 10 by 1
tmp_Y = (np.random.rand(10,1)>0.35).astype(float)

#with gradient
tmp_J, tmp_theta = gradient(tmp_X, tmp_Y, np.zeros((3,1)), 1e-8, 700)
print("cost" + str(tmp_J))
print("weight vector " + str([round(t, 8) for t in np.squeeze(tmp_theta)]))

def extractFeatures(tweet, freqs, process_tweet=process_tweet):
    word_l = process_tweet(tweet)
    x = np.zeros(3)
    x[0] = 1
    
    for word in word_l:
        x[1] += freqs.get((word, 1), 0) #positive label
        x[2] += freqs.get((word, 0), 0) #neg label
    x = x[None, :]
    assert(x.shape == (1,3))
    return x

#first test on training data
tmp1 = extractFeatures(train_x[0], freqs)
print(tmp1)

X = np.zeros((len(train_x), 3)) #collects features x and stacks to matrix X

for i in range(len(train_x)):
    X[i, :]= extractFeatures(train_x[i], freqs)
    
Y = train_y #training labels for X

#apply gradient
J, theta = gradient(X, Y, np.zeros((3, 1)), 1e-9, 1500)

print("cost of training" + str(J))
print("vector weights" + str([round(t, 8) for t in np.squeeze(theta)]))

def predict(tweet, freqs, theta):
    x = extractFeatures(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    
    return y_pred

#testing output
for tweet in ['I am happy', 'I am sad', 'this sucks', 'this is great', 'great', 'great great']:
    print(str(tweet) + " === " + str(predict(tweet, freqs, theta)))

def testAccuracy (test_x, test_y, freqs, theta, predict=predict):
    y_hat = []
    for tweet in test_x:
        y_pred = predict(tweet, freqs, theta)
        
        if y_pred > 0.5: #good prediction
            y_hat.append(1.0)
        else:
            y_hat.append(0.0)
            
    accuracy = np.sum(y_hat==test_y.squeeze())/len(test_x)#merge into a 1d array with == to compare
    
    return accuracy

tmp_accuracy = testAccuracy(test_x, test_y, freqs, theta)
print("accuracy: " + str(tmp_accuracy))


test = 'this is a really good thing. I loved this thing but the ending was terrible..'
print(process_tweet(test))
y_hat = predict(test, freqs, theta)
print(y_hat)
if y_hat>0.5:
    print("POSITIVE")
else:
    print("NEGATIVE")


    
