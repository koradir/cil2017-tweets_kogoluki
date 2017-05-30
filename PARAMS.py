#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
provides parameters for the CNN_TweetClassifier

@author: koradir
"""

class CNN_PARAMS:
    suppress_output = True
    nof_iterations = 1
    dropout_keep_probability = 0.5
    learning_rate = 1e-4
    print_frequency = 1
    
    """IF YOU CHANGE ANYTHING BELOW THIS LINE, THROW AWAY THE SAVED NETWORK AND RETRAIN!"""
    
    dim_embeddings = 300  # length of embedding vectors
    batch_size = 10
    nof_classes = 2  # negative= 0, positive = 1
    gram_sizes = [3,4,5]  # e.g. look at 3-, 4- and 5-grams
    
    conv1_feature_count = 100
    
    SPP_dimensions = [4,2,1] # creates 672 values per sample
    SPP_output_shape = (batch_size,
                       sum(conv1_feature_count*[2**i for i in SPP_dimensions]) - conv1_feature_count
                       )
    
    nof_neurons = 1024
    