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
        # TODO:
        pass

if __name__ == '__main__':
    clf = CNN_BaselineClassifier()
    print("Hello World")