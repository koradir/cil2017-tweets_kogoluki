#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A classifier using the python Natural Language Toolkit (NLTK), based on 
http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/
"""

import nltk
from Classifier import Classifier
from TweetRepresenter import TweetRepresenter


class NLTK_Classifier(Classifier):
    def __init__(self,
                 vocab='vocab.pkl',  # saved dictionary (word -> word number)
                 
                 save_as='./ntlk_classifier.pkl',  # save to this file
                 saved_model='./ntlk_classifier.pkl',  # set to None to discard saved model
                 retrain = False
                 ):
        super().__init__(vocab)
        self._tr = TweetRepresenter(self._vocab)
        
        self._save_as = save_as
        
        # TODO: check if classifier can be restored and restore if so
        self._clf = None
        
    def train(self,*examples,encoding='utf8'):
        assert len(examples) == 2, 'only implemented binary classification'
        
        ex_reps = []
        for i in range(len(examples)):
            with open(examples[i],encoding=encoding) as fex:
                tweets = fex.readlines()
                ex_reps.append(self._tr.represent(tweets))
                
        assert len(ex_reps) == len(examples)
        
        return self._train(ex_reps)
    
    def _train(self,ex_reps):
        assert len(examples) == 2, 'only implemented binary classification'
        pass
    
    def test(self,*examples,encoding='utf8'):
        assert len(examples) == 2, 'only implemented binary classification'
        
        ex_reps = []
        for i in range(len(examples)):
            with open(examples[i],encoding=encoding) as fex:
                tweets = fex.readlines()
                ex_reps.append(self._tr.represent(tweets))
                
        assert len(ex_reps) == len(examples)
        
        return self._test(ex_reps)
    
    def _test(self,ex_reps):
        assert len(examples) == 2, 'only implemented binary classification'
        
        labelled_tweets = [(tweet,-1) for tweet in ex_reps[0]]  # negative tweets
        labelled_tweets.extend([(tweet,1) for tweet in ex_reps[1]])  # positive tweets
        
        def extract_features(tweet_rep):
            features = {f'contains({t})' : False for t in self._vocab.}
        
        
        pass
    
    def predict(self,tweets,remap=None):
        predictions = self._predict(self._tr.represent(tweets))
        if remap is not None:
            for i,p in enumerate(predictions):
                r = remap.get(p,p)
                if r != p:
                    predictions[i] = r
                    
        return predictions
        
    def _predict(self,tweet_reps):
        return [self._prediction(t) for t in tweet_reps]
    
    def _prediction(self,tweet_rep,epsilon=0):
        pass
        
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
            ex_reps.append(clf._tr.represent(tweets))
    
    off0 = max([1,int(cv_frac * len(ex_reps[0]))])
    off1 = max([1,int(cv_frac * len(ex_reps[0]))])
    
    ex_reps_test = [ex_reps[0][:off0], ex_reps[1][:off1]]
    ex_reps_train = [ex_reps[0][off0:], ex_reps[1][off1:]]
    
    print("TRAINING")
    
    clf._train(ex_reps_train)
    
    print("TESTING")
    clf = NLTK_Classifier()
    acc = clf._test(ex_reps_train)
    print(f'accuracy(training set) = {acc}')
    
    acc = clf._test(ex_reps_test)
    print(f'accuracy(testing set) = {acc}')