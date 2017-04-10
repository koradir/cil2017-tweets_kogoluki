from tweet_svm import SVM_TweetClassifier
from tweet_cluster import ClusterClassifier
import csv

test='twitter-datasets/test_data.txt'
header=['Id','Prediction']
out='submission_clustered.csv'

#clf = SVM_TweetClassifier()
clf = ClusterClassifier()

with open(test,mode='r',encoding="utf8",newline='') as testfile:
    ## not this time ##next(testfile)  # skip header
    
    print('predicting ...')
    tweets = [t.split(',',1) for t in testfile.readlines()]
    tweets = [(i,clf.predict(t)) for i,t in tweets]
    
    print('writing...')
    with open(out,mode='w',encoding="utf8",newline='') as outfile:
        w = csv.writer(outfile)
        w.writerow(header)
        w.writerows(tweets)
        
    print('done')
            