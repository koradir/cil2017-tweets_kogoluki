#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline CNN implementation
"""

import tensorflow as tf
import numpy as np
import numpy.random as nprand

import PARAMS as par
from statusbar import status_update
from CNN_Classifier import CNN_Classifier

class CNN_BaselineClassifier(CNN_Classifier):
    def __init__(self,
                 vocab='vocab.pkl',  # saved dictionary (word -> word number)
                 embeddings=None,
                 save_as='./cnn_classifier.ckpt',  # save to this file
                 saved_model='./cnn_classifier.ckpt',  # set to None to discard saved model
                 debug=False,
                 retrain=False,
                 PARAMS=par.CNN_BASE_PARAMS
                 ):
        super().__init__(vocab=vocab,
                       embeddings=embeddings,
                       save_as=save_as,
                       saved_model=saved_model,
                       debug=debug,
                       retrain=retrain,
                       PARAMS=PARAMS
                       )
        
    def _build_model(self):
        PARAMS = self.PARAMS
        
        def conv2d(x,W,strides=[1,1,1,1],padding='VALID'):
            """note: by using 'VALID' padding, the convolution window `W`
            stays within the confines of `x`"""
            return tf.nn.conv2d(x,W,strides=strides,padding=padding)
        
        def max_pool(h,ksize=[1,-1,1,1],strides=[1,1,1,1],padding='VALID'):
            """note: by using 'VALID' padding, the pooling window stays
            within the confines of `h`"""
            return tf.nn.max_pool(h,ksize=ksize,strides=strides,padding=padding)
            
        
        self._x_input = tf.placeholder(tf.int32, shape=[PARAMS.batch_size, None])  # lists of tokens
        self._y_input = tf.placeholder(tf.int32, shape=[PARAMS.batch_size, PARAMS.nof_classes])  # expect one-hot
        
        """EMBEDDINGS"""
        # create on extra vector for padding
        embeddings = self._embedding_variable(
                shape=(len(self._vocab) + 1, self.PARAMS.dim_embeddings),
                name='embeddings')
        
        h_embed = tf.nn.embedding_lookup(embeddings,self._x_input)
        
        # reshape because conv2d expects 4d-tensors
        h_embed = tf.reshape(h_embed,[PARAMS.batch_size,-1,PARAMS.dim_embeddings,1])
        
        if self.debug:
            print("h_embed:",h_embed.get_shape())
            print()
        
        """CONVOLUTION"""
        pooled = []
        for f in PARAMS.gram_sizes:
            W_conv1f = self._weight_variable([f,PARAMS.dim_embeddings,1,PARAMS.conv1_feature_count],f'W_conv1_{f}')
            b_conv1f = self._bias_variable([PARAMS.conv1_feature_count],f'b_conv1_{f}')
            h_conv1f = tf.nn.relu(conv2d(h_embed,W_conv1f) + b_conv1f)
            
            if self.debug:
                print(f'h_conv1_{f}:',h_conv1f.get_shape())
                
            h_pool1f = max_pool(h_conv1f)
            pooled.append(h_pool1f)
            
            if self.debug:
                print(f'h_pool1_{f}:',h_pool1f.get_shape())
                print()
                
        h_pooled1 = tf.concat(pooled,axis=3)
        
        # flatten for fully connected layers
        nof_filters = len(PARAMS.gram_sizes)*PARAMS.conv1_feature_count
        h_flattened1 = tf.reshape(h_pooled1,shape=[-1,nof_filters])
                
        if self.debug:
            print('h_pooled1:',h_pooled1.get_shape())
            print('h_flattened1:',h_flattened1.get_shape())
            print()
            
        """FULLY CONNECTED LAYERS"""
        
        # dropout disables fraction of neurons ==> prevents co-adapting
        # (i.e. the activeneed to learn on its own)
        
        self._keep_prob = tf.placeholder(tf.float32)
        h_dropout1 = tf.nn.dropout(h_flattened1,self._keep_prob)
        
        if self.debug:
            print('h_drouput1:',h_dropout1.get_shape())
        
        # output
        W_fc1 = self._weight_variable([nof_filters,PARAMS.nof_classes],'W_fc1')
        b_fc1 = self._bias_variable([PARAMS.nof_classes],'b_fc1')
        h_fc1 = tf.nn.xw_plus_b(h_dropout1,W_fc1,b_fc1)
        
        if self.debug:
            print('h_fc1',h_fc1.get_shape())
        
        """MODEL OUTPUT"""
        y = h_fc1
        self._model = y
        
        if self.debug:
            print('y',y.get_shape())
            print()
        
        """LOSS & OPTIMISATION"""
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                        logits=y, labels=self._y_input )
                )
                
        self._train_step = tf.train.AdamOptimizer(PARAMS.learning_rate).minimize(loss)
        
        """EVALUATION"""
        correct_predictions = tf.equal(tf.argmax(y,1), tf.argmax(self._y_input,1))
        self._nof_corrects = tf.reduce_sum(tf.cast(correct_predictions, "float"))

if __name__ == '__main__':
    clf = CNN_BaselineClassifier(debug=True)
    print("Hello World")