#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This approach starts with the embeddings from glove and then
attempts to learn refined embeddings by means of a CNN.

Architecture:
    embedding layer
    -> convolution -> ReLU -> max-pooling
    -> convolution -> ReLU -> spatial pyramid pooling
    -> fully connected layer
    -> fully connected layer
    -> dropout layer
    -> softmax

@author: koradir
"""

import tensorflow as tf
import numpy as np
import numpy.random as nprand

try:
   import cPickle as pickle
except:
   import pickle
   
import os

from SpatialPyramidPooling import spatial_pyramid_pool
from PARAMS import CNN_PARAMS as PARAMS
from statusbar import status_update


class CNN_TweetClassifier:
    
    def __init__(self,
                 vocab='vocab.pkl',  # saved dictionary (word -> word number)
                 save_as='cnn_classifier.ckpt',  # save to this file
                 saved_model='cnn_classifier.ckpt',  # set to None to discard saved model
                 debug=False
                 ):
        
        self.debug = debug
        self._save_as = save_as
        
        """LOAD VOCABULARY"""
        self.vocab = {}
        with open(vocab,'rb') as f:
            self.vocab = pickle.load(f)
        print("vocabulary loaded")
        
        """BUILD MODEL"""
        self.model = None
        self._train_step = None
        self._accuracy = None
        self._tf_variables = []
        
        '''closing a session does not get rid of legacy stuff'''
        tf.reset_default_graph()
        
        '''builds graph and initialises above variables'''
        self._model()
        
        """INITIALISE TF VARIABLES"""
        model_restorable = (
                saved_model is not None
                and os.path.exists(f'{saved_model}.index')
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
                saver.save(self._session,save_as,write_meta_graph=False)
                
        if self.debug:
            assert not self._session._closed
    
    def __del__(self):
        print("CALLING __del__")
        if not self._session._closed:
            self._session.close()
        
    def _model(self):
        '''Like in Kim's "Convolutional Neural Network for Sentence Classification",
           (http://www.aclweb.org/anthology/D14-1181),
           we use filters of different sizes to capture different 'n-grams'.
           
           Kim's approach is to extract from each filter a single feature
           by means of a max-over-time pooling operation.
           
           The reasoning behind this move is that "this pooling scheme naturally
           deals with variable sentence lengths"
           
           However, about a year after Kim's paper, Kaiming et al published their
           vision of what they call "Spatial Pyramid Pooling (SPP)",
           (https://arxiv.org/pdf/1406.4729.pdf),
           with the express intent on avoiding padding or otherwise distorting the
           samples.
           
           SPP yields a fixed-size output, independent of the input size, and
           therefore provides a different way to deal with variable sentence lengths - 
           one that does not require limiting ourselves to a single feature
           per filter.
        '''
        def conv2d(x,W):
            '''
            2D convolution, expects 4D input x and filter matrix W
            '''
            return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding ='SAME')
        
        def weight_variable(shape,name):
            '''
            Initialise weights with a small amount of noise for symmetry breaking
            '''
            initial = tf.truncated_normal(shape,stddev=0.1)
            v = tf.Variable(initial,name=name)
            self._tf_variables.append(v)
            return v
        
        def bias_variable(shape,name):
            '''
            Initialise biases slightly positive to avoid "dead" neurons.
            '''
            initial = tf.constant(0.1,shape=shape)
            v = tf.Variable(initial,name=name)
            self._tf_variables.append(v)
            return v
        
        def EmbeddingVariable(shape,name):
            initial = tf.truncated_normal(shape)
            v = tf.Variable(initial,name=name)
            self._tf_variables.append(v)
            return v
        
        self._x_input = tf.placeholder(tf.int32, shape=[PARAMS.batch_size, None])  # lists of tokens
        self._y_input = tf.placeholder(tf.int32, shape=[PARAMS.batch_size, PARAMS.nof_classes])  # expect one-hot
        
        """EMBEDDING"""
        embeddings = EmbeddingVariable([len(self.vocab), PARAMS.dim_embeddings],'embeddings')
        
        h_embed = tf.nn.embedding_lookup(embeddings,self._x_input)
        # note: h_embed has dimensions batch_size x sentence_length x dim_embeddings
        
        # reshaping because conv2d expects 4-dim tensors
        h_embed = tf.reshape(h_embed,[PARAMS.batch_size,-1,PARAMS.dim_embeddings,1])
        
        
        """CONVOLUTIONAL LAYERS"""
        filter_sizes = PARAMS.gram_sizes
        nof_features = PARAMS.conv1_feature_count
        pooled = []
        for f in filter_sizes:
            W_conv1f = weight_variable([f,f,1,nof_features],f'W_conv1_{f}')
            b_conv1f = bias_variable([nof_features],f'b_conv1_{f}')
            h_conv1f = tf.nn.relu(conv2d(h_embed,W_conv1f) + b_conv1f)
            h_spp1f = spatial_pyramid_pool(h_conv1f,PARAMS.SPP_dimensions)
            
            if self.debug:
                print(f'h_conv1_f={f}:', h_conv1f.get_shape())
                print(f'h_spp1_f={f}:', h_spp1f.get_shape())
                print(f'expected_f={f}:',PARAMS.SPP_output_shape)
                print('')
            
            pooled.append(h_spp1f)
        
        h_conv1 =  tf.concat(pooled,axis=1)
        
        if self.debug:
            print('h_conv1:',h_conv1.get_shape())
            assert h_conv1.get_shape() == (
                    PARAMS.batch_size,
                    len(PARAMS.gram_sizes) * PARAMS.SPP_output_shape[1]
                    )
            
            
        """FULLY-CONNECTED LAYER with dropout"""
        nof_inputs = len(PARAMS.gram_sizes) * PARAMS.SPP_output_shape[1]
        
        W_fc1 = weight_variable([nof_inputs,PARAMS.nof_neurons],'W_fc1')
        b_fc1 = bias_variable([PARAMS.nof_neurons],'b_fc1')
        
        h_fc1 = tf.nn.relu(tf.matmul(h_conv1,W_fc1) + b_fc1)
        
        if self.debug:
            print('h_fc1:',h_fc1.get_shape())
        
        self._keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=self._keep_prob)
        
        
        """READOUT LAYER"""
        W_fc2 = weight_variable([PARAMS.nof_neurons,PARAMS.nof_classes],'W_fc2')
        b_fc2 = bias_variable([PARAMS.nof_classes],'b_fc2')
        
        h_fc2 = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
        '''note to self: !! NO RELU HERE !!'''
        
        if self.debug:
            print('h_fc2:',h_fc2.get_shape())
        
        
        """MODEL OUTPUT"""
        y = h_fc2
        self._model = tf.nn.softmax(y)
        
        if self.debug:
            print('y:',y.get_shape())
            assert y.get_shape() == (
                    PARAMS.batch_size,
                    PARAMS.nof_classes
                    )
            print('')
        
        
        """LOSS"""
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                        logits=y, labels=self._y_input )
                )
        
#        loss = tf.reduce_mean(
#                tf.nn.sigmoid_cross_entropy_with_logits(
#                        logits=y,labels=tf.cast(self._y_input,tf.float32)
#                        )
#                )
        
        self._train_step = tf.train.AdamOptimizer(PARAMS.learning_rate).minimize(loss)
        
        """ACCURACY"""
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(self._y_input,1))
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    def _representation(self,tweet):
        tokens = [self.vocab.get(t, -1) for t in tweet.strip().split()]
        tokens = [t for t in tokens if t >= 0]
        return tokens
    
    def train(self,*examples,encoding="utf8"):
        if len(examples) != PARAMS.nof_classes:
            print(f'ERR: expected {PARAMS.nof_classes} classes, got examples for {len(examples)}')
            return
        
        tweets = []
        for i in range(len(examples)):
            if self.debug:
                print(f'reading examples for class {i}')
            one_hot = np.zeros([PARAMS.nof_classes])
            one_hot[i] = 1
            
            with open(examples[i], encoding=encoding) as fpos:
                tweets_i = fpos.readlines()
            
            tweets_i_data = zip([self._representation(tweet) for tweet in tweets_i],
                            [one_hot] * len(examples[i])
                            )
            
            tweets += tweets_i_data
        
        '''
        CNN expects all examples in a batch to have the same length
        '''
        tdict = {}
        for t in tweets:
            tdict.setdefault(len(t[0]), []).append(t)

        def next_batch(ex_len,amount=PARAMS.batch_size):
            exs = tdict[ex_len]            
            idxs = nprand.randint(0,len(exs),amount)
            return zip(*[exs[i] for i in idxs ])
        
        """TRAINING"""
        with self._session.as_default():
            for it in range(PARAMS.nof_iterations):
                if it % PARAMS.print_frequency == 0:
                    k = nprand.randint(0,len(tdict.keys()))
                    k = list(tdict.keys())[k]
                    xs,ys = next_batch(k)
                    feed_dict = {self._x_input:xs,
                                 self._y_input:ys,
                                 self._keep_prob:1
                                 }
                    accuracy = self._accuracy.eval(feed_dict=feed_dict)
                    print(f'iteration {it}; train acc = {accuracy}')
                    
                    if self.debug:
                        output = self._session.run(self._model,feed_dict=feed_dict)
                        print('expected',ys)
                        print('got',output)
                        
                    
                top = len(list(tdict.keys())) -1
                curr = 0
                for i in tdict.keys():
                    status_update(curr,top,label=f'training size {i}')
                    curr += 1
                    xs,ys = next_batch(i)
                    feed_dict = {self._x_input:xs,
                                 self._y_input:ys,
                                 self._keep_prob:PARAMS.dropout_keep_probability
                                 }
                    self._session.run(self._train_step,feed_dict=feed_dict)
                    
                print('saving ...')
                saver = tf.train.Saver(self._tf_variables)
                saver.save(self._session,self._save_as,write_meta_graph=False)
    
    def test(self,tweets,classifications):
        pass
    
    def predict(self):
        pass
    
if __name__ == '__main__':
    datafolder = 'twitter-datasets'
    train_pos = f'{datafolder}/train_pos.txt'
    train_neg = f'{datafolder}/train_neg.txt'
#    train_pos = f'{datafolder}/train_pos_full.txt'
#    train_neg = f'{datafolder}/train_neg_full.txt'
    clf = CNN_TweetClassifier(debug=False)
    
    print("STARTING TRAINING")
    # class 0 if negative, class 1 if positive
    clf.train(train_neg,train_pos)
