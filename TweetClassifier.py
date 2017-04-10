import numpy as np
from pathlib import Path
try:
   import cPickle as pickle
except:
   import pickle
   

class TweetClassifier:
    
    clf_path = Path('svc_classifier.pkl')
    
    def __init__(self,vocab='vocab.pkl',embeddingsX='embeddingsX_K200_step0.001_epochs10.npy',
                 debug=False):
        
        self.debug = debug
        
        with open(vocab,'rb') as f:
            self.vocab = pickle.load(f)
        print("vocabulary loaded")
        # note: is dictonary (word -> word number)
        
        self.X = np.load(embeddingsX)
        
        self._clf = None
    
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
        pass
    
    def accuracy(self,pos,neg,encoding="utf8"):
        pass
    
    def predict(self,tweet):
        pass

    def _store_clf(self):  #store for later use
        if self._clf is None:
            return
        
        print("saving clf...")
        with open(self.clf_path,'wb') as fout:
            pickle.dump(self._clf,fout)
        print("saved")
        
    def _load_clf(self):
        if not self.clf_path.exists:
            print("the classifier needs first be trained")
            return False
        
        print("loading trained classifier...")
        with open(self.clf_path,'rb') as fin:
            self._clf = pickle.load(fin)
            
        print("loaded")
        return True