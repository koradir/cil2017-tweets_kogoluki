import numpy as np
from sklearn import preprocessing,svm
from pathlib import Path
from TweetClassifier import TweetClassifier

class SVM_TweetClassifier(TweetClassifier):
    
    clf_path = Path('svm_classifier.pkl')
    
    def __init__(self,vocab='vocab.pkl',embeddingsX='embeddingsX_K200_step0.001_epochs10.npy',
                 debug=False):
        
        super().__init__(vocab,embeddingsX,debug)
    
    def train(self,pos,neg,encoding="utf8"):
        train_x,train_y = self._load_training_data(pos,neg,encoding=encoding)
        
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
        
        test_x,test_y = self._load_training_data(pos,neg,encoding=encoding)
        
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
















    