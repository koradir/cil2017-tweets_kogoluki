#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
creates sample submission

@author: koradir
"""

from CNN_BaselineClassifier import CNN_BaselineClassifier
import csv


datafolder = 'twitter-datasets'
testfile = f'{datafolder}/test_data.txt'
outfile = 'submission.csv'
header = ['Id','Prediction']

with open(testfile,'r',encoding='utf-8',newline='') as ftest:
    lines = ftest.readlines()
    
lines = [ (int(x[0]),x[1]) for x in [s.split(',',1) for s in lines] ]

idxs,tweets = zip(*lines)

clf = CNN_BaselineClassifier(retrain=True)
    
print("TRAINING")
train_neg = f'{datafolder}/train_neg_full.txt'
train_pos = f'{datafolder}/train_pos_full.txt'
clf.train(train_neg,train_pos)

print("TESTING")
clf = CNN_BaselineClassifier()
acc = clf.test(train_neg,train_pos)
print(f'accuracy(training set) = {acc}')

print("PREDICTING")
output_results = clf.predict(tweets,remap={0:-1})

assert all(x == -1 or x == 1 for x in output_results), output_results

print(f'received {len(output_results)} predictions')

results = zip(idxs,output_results)

with open(outfile,mode='w',newline='') as fout:
    w = csv.writer(fout)
    w.writerow(header)
    for x in results:
        w.writerow(x)