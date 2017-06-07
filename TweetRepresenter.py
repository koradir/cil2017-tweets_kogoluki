#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 08:14:54 2017

@author: koradir
"""

import numpy as np

class TweetRepresenter:
    def __init__(self,vocab):
        self._vocab = vocab
        
    def _representation(self,tweet):
        """simple auxiliary method, looks up tokens in the provided vocabulary
           words not in the vocabulary are ignored
           note that we are adding +1 to all tokens s.t. we can zero-pad with
           impunity
        """
        
        tokens = [self.vocab.get(t, -1) + 1 for t in tweet.strip().split()]
        tokens = [t for t in tokens if t > 0]
        return tokens
    
    def represent(self,tweets):
        return [self._representation(tweet) for tweet in tweets]
    
    def represent_training_data(self,*examples,nof_classes,encoding='utf8',pad_with=None):
        """
        create a (representation, one_hot) tuple for each set of example)
        the one-hot is a vector of length `len(examples)` with one_hot[i]=1
        iff the representation is from a tweet out of `examples[i]`
        
        Parameters
        ----------
        examples : list of strings
            each string contains a path to a file, with
            `examples[i]` containing examples for class i
            
        nof_classes: int
            the number of classes to expect
            only used to verify enough example files have been provided
            
        pad_with : [int]
            if not None, all tweets are padded to the same length, using `pad_with`
            as a filler
        """
            
        assert len(examples) == nof_classes, f'ERR: expected {nof_classes} classes, got examples for {len(examples)}'
            
        tweets = []
        for i in range(len(examples)):
            if self.debug:
                print(f'reading examples for class {i}')
            one_hot = np.zeros([nof_classes])
            one_hot[i] = 1
            
            with open(examples[i], encoding=encoding) as fpos:
                tweets_i = fpos.readlines()
            
            tweets_i_data = zip(self.represent(tweets_i),
                            [one_hot] * len(examples[i])
                            )
            
            tweets += tweets_i_data
            
        if pad_with is not None:
            ts,ohs = zip(*tweets)
            len_max = max([len(t) for t in ts])
            tweets = zip([np.pad(t,pad_with=(0,len_max-len(t)),mode='constant',constant_values=pad_with) for t in ts],ohs)
            
        return tweets