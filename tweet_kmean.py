import numpy as np
from pathlib import Path
from TweetClassifier import TweetClassifier
from kmeans import kmeans
import itertools

class KmeansClassifier(TweetClassifier):
    
    clf_path = Path('kmeans_classifier.pkl')
    __cent_pos = None
    __cent_neg = None
    __centroids = None
    
    def __init__(self,
                 vocab='vocab.pkl',
                 embeddingsX='embeddingsX_K200_step0.001_epochs10.npy',
                 debug=False):
        
        super().__init__(vocab,embeddingsX,debug)
        
    
    def train(self,pos,neg,k=7,nofIterations=20,encoding="utf8"): 
        with open(pos, encoding=encoding) as fpos:
            tweets_pos = fpos.readlines()
            
        with open(neg, encoding=encoding) as fneg:
            tweets_neg = fneg.readlines()
        
        print("representing training data...")
        tweets_pos = np.array([self.representation(tweet) for tweet in tweets_pos])
        tweets_neg = np.array([self.representation(tweet) for tweet in tweets_neg])
        
        print("fitting...")
        self.__cent_pos = kmeans(tweets_pos,k=k,iterationCount=nofIterations)
        self.__cent_neg = kmeans(tweets_neg,k=k,iterationCount=nofIterations)
        
        ntotal = len(tweets_pos) + len(tweets_neg)
        ncorrect = 0
        
        self._clf = (self.__cent_pos,self.__cent_neg)
        
        self._store_clf()
        
        for t in tweets_pos:
            if self._predict(t) == 1:
                ncorrect += 1
                
        for t in tweets_neg:
            if self._predict(t) == -1:
                ncorrect += 1
                
        accuracy = ncorrect / ntotal
        print("classifier trained")
        print(f"accuracy on training set:{accuracy}")
        
    def _load_clf(self):
        super()._load_clf()
        self.__cent_pos=self._clf[0]
        self.__cent_neg=self._clf[1]
        return True
    
    def accuracy(self,pos,neg,encoding="utf8"): 
        if self._clf is None:
            if not self._load_clf():
                return -1
            
        with open(pos, encoding=encoding) as fpos:
            tweets_pos = fpos.readlines()
            
        with open(neg, encoding=encoding) as fneg:
            tweets_neg = fneg.readlines()
            
        tweets_pos = [self.representation(tweet) for tweet in tweets_pos]
        tweets_neg = [self.representation(tweet) for tweet in tweets_neg]
        
        ntotal = len(tweets_pos) + len(tweets_neg)
        ncorrect = 0
        
        for t in tweets_pos:
            if self._predict(t) == 1:
                ncorrect += 1
                
        for t in tweets_neg:
            if self._predict(t) == -1:
                ncorrect += 1
                
        return ncorrect / ntotal
    
    def predict(self,tweet):
        if self._clf is None:
            if not self._load_clf():
                return np.NaN
            else:
                self.__cent_pos, self.__cent_neg = self._clf
        
        rep = self.representation(tweet)
        
        return self._predict(rep)
    
    def _predict(self,rep):
#        dist_pos = np.min([np.linalg.norm(rep - c) for c in self.__cent_pos])
#        dist_neg = np.min([np.linalg.norm(rep - c) for c in self.__cent_neg])

        if self.__centroids is None:
            centroids = list(itertools.chain(
                    [(c,1) for c in self.__cent_pos],
                    [(c,-1) for c in self.__cent_neg]
                    ))
            
        centroids.sort(key=lambda x: np.linalg.norm(rep-x[0]))
        
        dist = np.sum([x[1] for x in centroids[:5]])
        
        if dist > 0:
            return 1
        elif dist < 0:
            return -1
        else:
            if np.random.randint(2) > 0:
                return 1
            else:
                return -1
    
if __name__ == '__main__':
    datafolder = 'twitter-datasets'
    train_pos = f'{datafolder}/train_pos.txt'
    train_neg = f'{datafolder}/train_neg.txt'
#    train_pos = f'{datafolder}/train_pos_full.txt'
#    train_neg = f'{datafolder}/train_neg_full.txt'
    clf = KmeansClassifier(embeddingsX='embeddingsX_K200_step0.001_epochs10.npy')
    #clf.train(train_pos,train_neg,k=20)
    print("accuracy on training set:",clf.accuracy(train_pos,train_neg))