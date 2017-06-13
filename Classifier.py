#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:51:20 2017

@author: koradir
"""


try:
   import cPickle as pickle
except:
   import pickle
   

class Classifier:
    
    def __init__(self,
                 vocab='vocab.pkl',  # saved dictionary (word -> word number)
                 ):
        """LOAD VOCABULARY"""
        with open(vocab,'rb') as f:
            self._vocab = pickle.load(f)
            
    def train(self,*examples,encoding='utf8'):  
        pass
    
    def test(self,*examples,encoding='utf8'):
        pass
    
    def predict(self,tweets,remap=None):
        pass