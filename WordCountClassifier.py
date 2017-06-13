#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A very simple approach at classifying that looks at how often each word - individually - has been 
"""
import os
import pickle
import numpy as np

from Classifier import Classifier
from TweetRepresenter import TweetRepresenter


class WordCountClassifier(Classifier):
    
    def __init__(self,
                 vocab='vocab.pkl',  # saved dictionary (word -> word number)
                 
                 save_as='./wc_classifier.pkl',  # save to this file
                 saved_model='./wc_classifier.pkl',  # set to None to discard saved model
                 retrain = False
                 ):
        super().__init__(vocab)
        
        self._tr = TweetRepresenter(self._vocab)
        self._save_as = save_as
        
        restorable = os.path.exists(f'{saved_model}.index') and not retrain

        if restorable:
            with open(saved_model,'rb') as f:
                self._clf = pickle.load(f)
        else:
            self._clf = {}
        
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
        assert len(ex_reps) == 2, 'only implemented binary classification'      
        for tweet in ex_reps[0]:
            for t in tweet:
                self._clf[t] = self._clf.get(t,0) -1
                
        for tweet in ex_reps[0]:
            for t in tweet:
                self._clf[t] = self._clf.get(t,0) +1
                
        with open(self._save_as,'wb') as fout:
            pickle.dump(self._clf,fout)
        
    
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
        assert len(ex_reps) == 2, 'only implemented binary classification'
        
        nof_corrects = 0
        nof_examples = 0
        
        ps,counts = np.unique(self._predict(ex_reps[0]), return_counts=True)
        nof_corrects += dict(zip(ps,counts)).get(-1,0)
        nof_examples += len(ex_reps[0])
        
        ps,counts = np.unique(self._predict(ex_reps[1]), return_counts=True)
        nof_corrects += dict(zip(ps,counts)).get(1,0)
        nof_examples += len(ex_reps[1])
        
        return nof_corrects / nof_examples
        
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
        c = 0
        for t in tweet_rep:
            c += self._clf.get(t,0)
            
        if c < 0 - epsilon:
            return -1
        elif c > 0 + epsilon:
            return 1
        else:
            if np.random.randint(2) > 0:
                return 1
            else:
                return -1
            

if __name__ == '__main__':
    clf = WordCountClassifier(retrain=True)
    datafolder = 'twitter-datasets'
    train_neg = f'{datafolder}/train_neg.txt'
    train_pos = f'{datafolder}/train_pos.txt'
    
    print("TRAINING")
    clf.train(train_neg,train_pos)
    
    print("TESTING")
    clf = WordCountClassifier()
    acc = clf.test(train_neg,train_pos)
    print(f'accuracy(training set) = {acc}')
    
    train_neg = f'{datafolder}/train_neg_full.txt'
    train_pos = f'{datafolder}/train_pos_full.txt'
    
    acc = clf.test(train_neg,train_pos)
    print(f'accuracy(testing set) = {acc}')