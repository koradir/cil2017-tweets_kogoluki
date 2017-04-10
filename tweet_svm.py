import numpy as np
from sklearn import preprocessing,svm
from pathlib import Path
from TweetClassifier import TweetClassifier

class SVM_TweetClassifier(TweetClassifier):
    
    clf_path = Path('svc_classifier.pkl')
    svc_kernel = 'linear'
    
    def __init__(self,vocab='vocab.pkl',embeddingsX='embeddingsX_K200_step0.001_epochs10.npy',
                 debug=False):
        
        super().__init__(vocab,embeddingsX,debug)
    
    def train(self,pos,neg,encoding="utf8"):
        if not self.debug:
            with open(pos, encoding=encoding) as fpos:
                tweets_pos = fpos.readlines()
                
            with open(neg, encoding=encoding) as fneg:
                tweets_neg = fneg.readlines()
        else:
            tweets_pos = "Hello World", "I'm fine", "Of course!".lower()
            tweets_neg = "Yeah, sure ...", "Sorry", "If you must...".lower()
            
        print("representing training data...")
        tweets_pos = [self.representation(tweet) for tweet in tweets_pos]
        results_pos = [1]*len(tweets_pos)
        tweets_neg = [self.representation(tweet) for tweet in tweets_neg]
        results_neg = [-1]*len(tweets_neg)
        
        train_x = np.concatenate((tweets_pos,tweets_neg))
        train_y = np.concatenate((results_pos,results_neg))
        
        #classifier
        self._clf = svm.LinearSVC(dual=False)
    
        print("fitting...")
        self._clf.fit(train_x,train_y)
        
        #store for later use
        self._store_clf()
        
        accuracy = self._clf.score(train_x,train_y)
        print("classifier trained")
        print(f"accuracy on training set:{accuracy}")
        
    def accuracy(self,pos,neg,encoding="utf8"):
        print("ACCURACY")
        if self._clf is None:
            if not self._load_clf():
                return -1
        
        with open(pos, encoding=encoding) as fpos:
            tweets_pos = fpos.readlines()
            
        with open(neg, encoding=encoding) as fneg:
            tweets_neg = fneg.readlines()
        
        print("representing tweets ...")
        tweets_pos = [self.representation(tweet) for tweet in tweets_pos]
        results_pos = [1]*len(tweets_pos)
        tweets_neg = [self.representation(tweet) for tweet in tweets_neg]
        results_neg = [-1]*len(tweets_neg)
        
        print("testing ...")
        test_x = np.concatenate((tweets_pos,tweets_neg))
        test_y = np.concatenate((results_pos,results_neg))
        
        return self._clf.score(test_x,test_y)
        
    def predict(self,tweet):
        if self._clf is None:
            if not self._load_clf():
                return np.NaN
                
        return self._clf.predict(self.representation(tweet).reshape(1,-1))

if __name__ == '__main__':
    datafolder = 'twitter-datasets'
    #train_pos = f'{datafolder}/train_pos.txt'
    #train_neg = f'{datafolder}/train_neg.txt'
    train_pos = f'{datafolder}/train_pos_full.txt'
    train_neg = f'{datafolder}/train_neg_full.txt'
    clf = SVM_TweetClassifier(embeddingsX='embeddingsX_K200_step0.001_epochs10.npy')
    clf.train(train_pos,train_neg)
















    