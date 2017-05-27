# -*- coding: utf-8 -*-
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle
from MatrixPlotter import MatrixPlotter

datafolder = 'twitter-datasets'
train_neg = f'{datafolder}/train_neg.txt'
train_pos = f'{datafolder}/train_pos.txt'
#train_neg = f'{datafolder}/train_neg_full.txt'
#train_pos = f'{datafolder}/train_pos_full.txt'
encoding='utf8'

vocabulary='vocab.pkl'
embeddingsX='embeddingsX_K200_step0.001_epochs10.npy'
   
with open(vocabulary,'rb') as f:
    vocab = pickle.load(f)
    
X = np.load(embeddingsX)

def representation(tweet):        
    K = X.shape[1]
    
    def tokenise():
        tokens = [vocab.get(t, -1) for t in tweet.strip().split()]
        tokens = [t for t in tokens if t >= 0]
        return tokens
    
    tokens = tokenise()
    c = len(tokens)
    rep = np.zeros(K) + np.sum([X[i,:] for i in tokens],axis=0)
    
    if c > 1:
        rep /= c
    
    return rep

with open(train_pos, encoding=encoding) as fpos:
    tweets_pos = fpos.readlines()
    
with open(train_neg, encoding=encoding) as fneg:
    tweets_neg = fneg.readlines()

tweets_pos = [representation(tweet) for tweet in tweets_pos]
tweets_neg = [representation(tweet) for tweet in tweets_neg]

'''tweets as ROW vectors ==> transpose for plotting'''
mp = MatrixPlotter()
mp.plot(np.array(tweets_pos).T,np.array(tweets_neg).T,labels=['positive','negative'])