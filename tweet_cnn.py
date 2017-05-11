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

try:
   import cPickle as pickle
except:
   import pickle

from SpatialPyramidPooling import spatial_pyramid_pool
from pathlib import Path
from PARAMS import CNN_PARAMS as PARAMS


class CNN_TweetClassifier:
    
    clf_path = Path('cnn_classifier.pkl')
    
    def __init__(self,vocab='vocab.pkl',debug=False):
        
        self.debug = debug
        
        self.vocab = {}
        with open(vocab,'rb') as f:
            self.vocab = pickle.load(f)
        print("vocabulary loaded")
        # note: is a dictonary (word -> word number)        
        
        self._clf = None
        
        self._tf_train_step = None
        self._tf_accuracy = None
        
        self.model = self._model()
        
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
        
        def weight_variable(shape):
            '''
            Initialise weights with a small amount of noise for symmetry breaking
            '''
            initial = tf.truncated_normal(shape,stddev=0.1)
            return tf.Variable(initial)
        
        def bias_variable(shape):
            '''
            Initialise biases slightly positive to avoid "dead" neurons.
            '''
            initial = tf.constant(0.1,shape=shape)
            return tf.Variable(initial)
        
        def EmbeddingVariable(shape):
            initial = tf.truncated_normal(shape)
            return tf.Variable(initial)
        
        x_input = tf.placeholder(tf.int32, shape=[PARAMS.batch_size, None])  # lists of tokens
        y_input = tf.placeholder(tf.int32, shape=[PARAMS.batch_size, 1])  # 1 for positive, 0 for negative
        
        """EMBEDDING"""
        embeddings = EmbeddingVariable([len(self.vocab), PARAMS.dim_embeddings])
        
        h_embed = tf.nn.embedding_lookup(embeddings,x_input)
        # note: h_embed has dimensions batch_size x sentence_length x dim_embeddings
        
        # reshaping because conv2d expects 4-dim tensors
        h_embed = tf.reshape(h_embed,[PARAMS.batch_size,-1,PARAMS.dim_embeddings,1])
        
        
        """CONVOLUTIONAL LAYERS"""
        filter_sizes = PARAMS.gram_sizes
        nof_features = PARAMS.conv1_feature_count
        pooled = []
        for f in filter_sizes:
            W_conv1f = weight_variable([f,f,1,nof_features])
            b_conv1f = bias_variable([nof_features])
            h_conv1f = tf.nn.relu(conv2d(h_embed,W_conv1f) + b_conv1f)
            h_spp1f = spatial_pyramid_pool(h_conv1f,PARAMS.SPP_dimensions)
            
            if(self.debug):
                print(f'h_conv1_f={f}:', h_conv1f.get_shape())
                print(f'h_spp1_f={f}:', h_spp1f.get_shape())
                print(f'expected_f={f}:',PARAMS.SPP_output_shape)
                print('')
            
            pooled.append(h_spp1f)
        
        h_conv1 =  tf.concat(pooled,axis=1)
        
        if(self.debug):
            print('h_conv1:',h_conv1.get_shape())
            assert h_conv1.get_shape() == (
                    PARAMS.batch_size,
                    len(PARAMS.gram_sizes) * PARAMS.SPP_output_shape[1]
                    )
            
        """FULLY-CONNECTED LAYER with dropout"""
        nof_inputs = len(PARAMS.gram_sizes) * PARAMS.SPP_output_shape[1]
        
        W_fc1 = weight_variable([nof_inputs,PARAMS.nof_neurons])
        b_fc1 = bias_variable([PARAMS.nof_neurons])
        
        h_fc1 = tf.nn.relu(tf.matmul(h_conv1,W_fc1) + b_fc1)
        
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=keep_prob)
        
        
        """READOUT LAYER"""
        W_fc2 = weight_variable([PARAMS.nof_neurons,PARAMS.nof_classes])
        b_fc2 = bias_variable([PARAMS.nof_classes])
        
        h_fc2 =tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
        
        return h_fc2
    
    
    
if __name__ == '__main__':
    datafolder = 'twitter-datasets'
    #train_pos = f'{datafolder}/train_pos.txt'
    #train_neg = f'{datafolder}/train_neg.txt'
    train_pos = f'{datafolder}/train_pos_full.txt'
    train_neg = f'{datafolder}/train_neg_full.txt'
    clf = CNN_TweetClassifier(debug=True)
    #clf.train(train_pos,train_neg)
