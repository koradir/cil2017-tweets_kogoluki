#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
creates sample submission

@author: koradir
"""

from tweet_cnn import CNN_TweetClassifier
from tweet_svm import SVM_TweetClassifier
import csv


datafolder = 'twitter-datasets'
testfile = f'{datafolder}/test_data.txt'
outfile = 'submission_svm2.csv'
header = ['Id','Prediction']

with open(testfile,'r',encoding='utf-8',newline='') as ftest:
    lines = ftest.readlines()
    
lines = [ (int(x[0]),x[1]) for x in [s.split(',',1) for s in lines] ]

idxs,tweets = zip(*lines)

#clf = CNN_TweetClassifier(embeddings='embeddingsX_K300_step0.001_epochs50.npy')
#output_results = clf.predict(tweets)
#
#print(f'received {len(output_results)} predictions')
#
#"""expected classification for "negative" is -1, not 0"""
#for i,item in enumerate(output_results):
#    if item == 0:
#        output_results[i] = -1

clf=SVM_TweetClassifier(embeddingsX='embeddingsX_K300_step0.001_epochs50.npy')
output_results = [clf.predict(t) for t in tweets]

results = zip(idxs,output_results)

with open(outfile,mode='w',newline='') as fout:
    w = csv.writer(fout)
    w.writerow(header)
    for x in results:
        w.writerow(x)