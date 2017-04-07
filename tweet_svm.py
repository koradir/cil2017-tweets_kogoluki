import numpy as np
from sklearn import preprocessing,svm
from pathlib import Path
try:
   import cPickle as pickle
except:
   import pickle
   
class TweetClassifier:
    
    clf_path = Path('svc_classifier.pkl')
    svc_kernel = 'linear'
    
    def __init__(self,vocab='vocab.pkl',embeddingsX='embeddingsX_K200_step0.001_epochs10.npy',
                 debug=False):
        
        self.debug = debug
        
        with open(vocab,'rb') as f:
            self.vocab = pickle.load(f)
        print("vocabulary loaded")
        # note: is dictonary (word -> word number)
        
        self.X = np.load(embeddingsX)
        
        self.__clf = None
    
    def representation(self,tweet):        
        K = self.X.shape[1]
        
        def tokenise():
            tokens = [self.vocab.get(t, -1) for t in tweet.strip().split()]
            tokens = [t for t in tokens if t >= 0]
            return tokens
        
        tokens = tokenise()
        c = len(tokens)
        rep = np.zeros(K) + np.sum([self.X[i,:] for i in tokens],axis=0)
        
        if c > 1:
            rep /= c
        
        return rep
    
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
        self.__clf = svm.LinearSVC(dual=False)
    
        print("fitting...")
        self.__clf.fit(train_x,train_y)
        
        #store for later use
        print("saving...")
        with open(self.clf_path,'wb') as fout:
            pickle.dump(self.__clf,fout)
        
        accuracy = self.__clf.score(train_x,train_y)
        print("classifier trained")
        print(f"accuracy on training set:{accuracy}")
        
    def __load_clf(self):
        if not self.clf_path.exists:
            print("the classifier needs first be trained")
            return False
        
        print("loading trained classifier...")
        with open(self.clf_path,'rb') as fin:
            self.__clf = pickle.load(fin)
            
        print("loaded")
        return True
        
    def accuracy(self,pos,neg,encoding="utf8"):
        print("ACCURACY")
        if self.__clf is None:
            if not self.__load_clf():
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
        
        return self.__clf.score(test_x,test_y)
        
    def predict(self,tweet):
        if self.__clf is None:
            if not self.__load_clf():
                return np.NaN
                
        return self.__clf.predict(self.representation(tweet).reshape(1,-1))

if __name__ == '__main__':
    datafolder = 'twitter-datasets'
    #train_pos = f'{datafolder}/train_pos.txt'
    #train_neg = f'{datafolder}/train_neg.txt'
    train_pos = f'{datafolder}/train_pos_full.txt'
    train_neg = f'{datafolder}/train_neg_full.txt'
    clf = TweetClassifier(embeddingsX='embeddingsX_K200_step0.001_epochs10.npy')
    clf.train(train_pos,train_neg)
















    