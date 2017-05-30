import numpy as np
from pathlib import Path
from TweetClassifier import TweetClassifier

class ClusterClassifier(TweetClassifier):
    
    clf_path = Path('cluster_classifier.pkl')
    __cent_pos = None
    __cent_neg = None
    
    def __init__(self,
                 vocab='vocab.pkl',
                 embeddingsX='embeddingsX_K200_step0.001_epochs10.npy',
                 debug=False):
        
        super().__init__(vocab,embeddingsX,debug)
        
    
    def train(self,pos,neg,encoding="utf8"): 
        with open(pos, encoding=encoding) as fpos:
            tweets_pos = fpos.readlines()
            
        with open(neg, encoding=encoding) as fneg:
            tweets_neg = fneg.readlines()
        
        print("representing training data...")
        tweets_pos = [self.representation(tweet) for tweet in tweets_pos]
        tweets_neg = [self.representation(tweet) for tweet in tweets_neg]
        
        print("fitting...")
        self.__cent_pos = np.average(tweets_pos,axis=0)
        self.__cent_neg = np.average(tweets_neg,axis=0)
        
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
            if self._predicit(t) == -1:
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
        dist_pos = np.linalg.norm(rep - self.__cent_pos)
        dist_neg = np.linalg.norm(rep - self.__cent_neg)
        
        if dist_pos < dist_neg:
            return 1
        elif dist_pos > dist_neg:
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
    clf = ClusterClassifier(embeddingsX='embeddingsX_K200_step0.001_epochs10.npy')
    clf.train(train_pos,train_neg)