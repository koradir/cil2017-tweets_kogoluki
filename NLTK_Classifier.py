#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A classifier using the python Natural Language Toolkit (NLTK), based on 
http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/
"""

import os
from timeit import default_timer as timer
import pickle
import nltk
import numpy as np
from Classifier import Classifier


class NLTK_Classifier(Classifier):
    
    _MIN_TOKEN_LEN = 3
    
    def __init__(self,
                 vocab='vocab.pkl',  # saved dictionary (word -> word number)
                 
                 save_as='./ntlk_classifier.pkl',  # save to this file
                 saved_model='./ntlk_classifier.pkl',  # set to None to discard saved model
                 retrain = False
                 ):
        super().__init__(vocab)
        
        self._word_list = [k for k in self._vocab.keys() if len(k) >= self._MIN_TOKEN_LEN]
        
        self._save_as = save_as
        
        restorable = os.path.exists(f'{saved_model}') and not retrain

        if restorable:
            with open(saved_model,'rb') as f:
                self._clf = pickle.load(f)
        else:
            self._clf = None
    
    def _represent(self,tweets,sentiment=None):
        if sentiment is not None:
            return [([t.lower() for t in tweet.strip().split() if len(t) >= self._MIN_TOKEN_LEN],sentiment) for tweet in tweets]
        else:
            return [[t.lower() for t in tweet.strip().split() if len(t) >= self._MIN_TOKEN_LEN] for tweet in tweets]
        
        
    def train(self,*examples,encoding='utf8'):
        assert len(examples) == 2, 'only implemented binary classification'
        
        tweets = []
        with open(examples[0],'r',encoding=encoding) as fneg:
            tweets.extend(self._represent(fneg.readlines(),'neg'))
        with open(examples[1],'r',encoding=encoding) as fpos:
            tweets.extend(self._represent(fpos.readlines(),'pos'))
            
        return self._train(tweets)
    
    def _extract_freatures(self,tweet):
        features = {f'contains({t})' : False for t in self._word_list}
        for t in tweet:
            features[f'contains({t})'] = True
        return features
        
    
    def _train(self,labelled_tweets):        
        training_set = nltk.classify.apply_features(self._extract_freatures,labelled_tweets)
        self._clf = nltk.NaiveBayesClassifier.train(training_set)
        
        with open(self._save_as,'wb') as fout:
            pickle.dump(self._clf,fout)
    
    def test(self,*examples,encoding='utf8'):
        assert len(examples) == 2, 'only implemented binary classification'
        assert self._clf is not None, 'need to train first'
        
        tweets = []
        with open(examples[0],'r',encoding=encoding) as fneg:
            tweets.extend(self._represent(fneg.readlines(),'neg'))
        with open(examples[1],'r',encoding=encoding) as fpos:
            tweets.extend(self._represent(fpos.readlines(),'pos'))
        
        return self._test(tweets)
    
    def _test(self,labelled_tweets):     
        tweets,expectations = zip(*labelled_tweets)
        predictions = self._predict(tweets)
        
        nof_corrects = np.sum([1 if a == b else 0 for a,b in zip(predictions,expectations)])
        nof_examples = len(tweets)
        
        return nof_corrects/nof_examples       
    
    def predict(self,tweets,remap={'neg':-1,'pos':1}):
        predictions = self._predict(self._represent(tweets))
        if remap is not None:
            for i,p in enumerate(predictions):
                r = remap.get(p,p)
                if r != p:
                    predictions[i] = r
                    
        return predictions
        
    def _predict(self,tweets):
        return [self._prediction(t) for t in tweets]
    
    def _prediction(self,tweet_rep):
        assert self._clf is not None, 'need to train first'
        return self._clf.classify(self._extract_freatures(tweet_rep))
        
if __name__ == '__main__':
    clf = NLTK_Classifier(retrain=True)
    datafolder = 'twitter-datasets'
    train_neg = f'{datafolder}/train_neg_full.txt'
    train_pos = f'{datafolder}/train_pos_full.txt'
    cv_frac = 0.1
    
    examples = [train_neg,train_pos]
    ex_reps = []
    for i in range(len(examples)):
        with open(examples[i],encoding='utf8') as fex:
            tweets = fex.readlines()
            ex_reps.append(tweets)
    
    off0 = max([1,int(cv_frac * len(ex_reps[0]))])
    off1 = max([1,int(cv_frac * len(ex_reps[0]))])
    
    ex_reps_test = clf._represent(ex_reps[0][:off0],'neg')
    ex_reps_test.extend(clf._represent(ex_reps[1][:off1],'pos'))
    
    ex_reps_train = clf._represent(ex_reps[0][off0:],'neg')
    ex_reps_train.extend(clf._represent(ex_reps[1][off1:],'pos'))

#    # debug: only test with small sets
#    ex_reps_test = clf._represent(ex_reps[0][:200],'neg')
#    ex_reps_test.extend(clf._represent(ex_reps[1][:200],'pos'))
#    
#    ex_reps_train = clf._represent(ex_reps[0][200:700],'neg')
#    ex_reps_train.extend(clf._represent(ex_reps[1][200:700],'pos'))
    
#    
#    print("TRAINING")
#    s = timer()
#    clf._train(ex_reps_train)
#    e = timer()
#    print(f'training completed in {e-s}s')
    
    print("TESTING")
    clf = NLTK_Classifier()
    s = timer()
    acc = clf._test(ex_reps_train)
    e = timer()
    print(f'accuracy(training set) = {acc}')
    print(f'(test run completed in {e-s}s)')
    
    
    s = timer()
    acc = clf._test(ex_reps_test)
    e = timer()
    print(f'accuracy(testing set) = {acc}')
    print(f'(test run completed in {e-s}s)')