#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import numpy.random as nprand

try:
   import cPickle as pickle
except:
   import pickle
   
import os

import PARAMS as par
from statusbar import status_update
from TweetRepresenter import TweetRepresenter

class CNN_Classifier:
    def __init__(self,
                 vocab='vocab.pkl',  # saved dictionary (word -> word number)
                 embeddings=None,
                 save_as='./cnn_classifier.ckpt',  # save to this file
                 saved_model='./cnn_classifier.ckpt',  # set to None to discard saved model
                 debug=False,
                 retrain=False,
                 PARAMS=par.CNN_BASE_PARAMS
                 ):
        
        self.PARAMS = PARAMS
        self.debug = debug
        self._save_as = save_as
        
        
        """LOAD VOCABULARY"""
        self._vocab = {}
        with open(vocab,'rb') as f:
            self._vocab = pickle.load(f)
        if not self.PARAMS.suppress_output:
            print("vocabulary loaded")
            
        self._tr = TweetRepresenter(self._vocab)
        
        """BUILD MODEL"""
        self._embeddings = None     # embeddings matrix
        self._x_input = None        # expect placeholder (tweet vectors)
        self._y_input = None        # expect placeholder (one-hot vectors)
        self._keep_prob = None     # expect placeholder (dropout keep probability)
        self._model = None          # model output
        self._train_step = None     # runnable training step
        self._nof_corrects = None   # nof correct predictions in the batch
        self._tf_variables = []     # variables to be saved/restored
        
        '''closing a session does not get rid of legacy stuff'''
        tf.reset_default_graph()
        
        '''builds graph and initialises above variables'''
        self._build_model()
        
        '''check if all members have been set'''
        assert self._embeddings is not None
        assert self._x_input is not None
        assert self._y_input is not None
        assert self._keep_prob is not None
        assert self._train_step is not None
        assert self._nof_corrects is not None
        assert self._tf_variables
        
        """INITIALISE TF VARIABLES"""
        model_restorable = (
                saved_model is not None
                and os.path.exists(f'{saved_model}.index')
                and not retrain
                )
            
        
        if self.debug:
            print('restore from file: {0}'.format('yes' if model_restorable else 'no'))
        
        self._session = tf.Session()
        
        saver = tf.train.Saver(self._tf_variables)
        with self._session.as_default():
            if model_restorable:
                tf.global_variables_initializer().run()
                saver.restore(self._session,saved_model)
            else:
                tf.global_variables_initializer().run()
                
                if embeddings is not None:
                    assert self._embeddings is not None, f'ERR: self._embeddings not set'
                    E = np.load(embeddings)
                    assert E.shape == self._embeddings.get_shape(), f'ERR: expected embedding matrix of size {self._embeddings.get_shape()} but got {E.shape}.'
                    tf.assign(self._embeddings,E).run()
                    self._save_clf()
                
        if self.debug:
            assert not self._session._closed
    
    def __del__(self):
        print("CALLING __del__")
        if not self._session._closed:
            self._session.close()
            
    def _save_clf(self):
        saver = tf.train.Saver(self._tf_variables)
        saver.save(self._session,self._save_as,write_meta_graph=False)
            
    def _get_representation(self,*examples,encoding):
        return self._tr.represent_training_data(
                *examples,
                nof_classes=self.PARAMS.nof_classes,
                encoding=encoding,
                pad_with=0
                )
        
    def train(self,*examples,encoding='utf8'):        
        return self._train(self._get_representation(*examples,encoding))
    
    def _next_batch(self,lst,start):
        batch_size = self.PARAMS.batch_size
        
        end = min([start + batch_size,len(lst)])
        
        batch = lst[start,end]
        
        missing = batch_size = len(batch)
        if missing > 0:
            idxs = nprand.randint(0,len(lst),missing)
            batch.extend(lst[idxs])
            
        return end, batch
        
    def _train(self,tweet_reps):            
        with_output = not self.PARAMS.suppress_output
        
        top = len(tweet_reps)
            
        for e in range(self.PARAMS.nof_iterations):
            i=0
            if with_output:
                label='epoch {:>2}'.format(e+1)
                status_update(i,top,label)
            
            nprand.shuffle(tweet_reps)
            while i < top:
                i, batch = self._next_batch(tweet_reps,i)
                xs,ys = zip(*batch)
                
                self._session.run(self._train_step,feed_dict = {
                        self._x_input : xs,
                        self._y_input : ys,
                        self._keep_probs : self.PARAMS.dropout_keep_probability
                        })
                self._save_clf()
                if with_output:
                    status_update(i,top,label)
            
            if with_output and e % self.PARAMS.print_frequency == 0:
                accuracy = self._test(tweet_reps)
                print(f'accuracy(training set) = {accuracy}')
    
    def test(self,*examples,encoding='utf8'):
        """calculates and returns prediction accuracy"""
        
        return self._test(self._get_representation(*examples,encoding))
    
    def _test(self,tweet_reps):            
        with_output = not self.PARAMS.suppress_output
        
        top = len(tweet_reps)
            
        i=0
        nof_hits = 0
        if with_output:
            label='calculating accuracy'
            status_update(i,top,label)
        
        while i < top:
            i, batch = self._next_batch(tweet_reps,i)
            xs,ys = zip(*batch)
            
            nof_hits += self._session.run(self._nof_corrects,feed_dict = {
                    self._x_input : xs,
                    self._y_input : ys,
                    self._keep_probs : 1
                    })
            if with_output:
                status_update(i,top,label)
                
        return nof_hits/top
    
    def predict(self,tweets,remap=None):
        predictions = self._predict(self._tr.represent(tweets))
        
        if remap is not None:
            for i,p in enumerate(predictions):
                m = {}
                r = m.get(p,p)
                if r != p:
                    predictions[i] = r
                    
        return predictions
    
    def _predict(self,tweet_reps):          
        with_output = not self.PARAMS.suppress_output
            
        i=0
        top = len(tweet_reps)
        predictions = []
        if with_output:
            label='predicting'
            status_update(i,top,label)
        
        while i < top:
            s = min([i+self.PARAMS.batch_size],top) - i
            
            i, xs = self._next_batch(tweet_reps,i)
            
            preds = self._session.run(
                tf.slice(tf.argmax(self._model,1),[0],[s]),
                feed_dict={
                        self._x_input:xs,
                        self._keep_prob:1
                })
            predictions.extend(preds)
            if with_output:
                status_update(i,top,label)
                
        assert len(predictions) == len(tweet_reps)
                
        return predictions
            
    def _build_model(self):
        """
        construct graph and initialise variables set to None in  the __init__
        """
        pass
    
    def _weight_variable(self,shape,name):
        '''
        Initialise weights with a small amount of noise for symmetry breaking
        '''
        initial = tf.truncated_normal(shape,stddev=0.1)
        v = tf.Variable(initial,name=name)
        self._tf_variables.append(v)
        return v
    
    def _bias_variable(self,shape,name):
        '''
        Initialise biases slightly positive to avoid "dead" neurons.
        '''
        initial = tf.constant(0.1,shape=shape)
        v = tf.Variable(initial,name=name)
        self._tf_variables.append(v)
        return v
    
    def _embedding_variable(self,shape,name,set_default=True):
        initial = tf.truncated_normal(shape)
        v = tf.Variable(initial,name=name)
        self._tf_variables.append(v)
        if set_default:
            self._embeddings = v
        return v