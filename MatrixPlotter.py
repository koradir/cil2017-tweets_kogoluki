# -*- coding: utf-8 -*-
"""
Takes a matrix and uses PCA to plot it to a two-dimensional space.
"""

import numpy as np
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['legend.loc'] = 'best'

class MatrixPlotter:
    
    _scatterplots = []
    
    def plot(self,*matrices,labels=None):
        """
        Assumes data is given as column vectors (transpose matrix if not).
        All matrices' columns must have the same length (but they can have
        different amount of columns).
        
        Uses PCA to reduce dimensions to two, then plots them.
        """
        assert labels is None or len(labels) == len(matrices)
        N = matrices[0].shape[0]
        assert all(m.shape[0] == N for m in matrices)        
    
        sizes = [m.shape[1] for m in matrices]
        X = np.concatenate(matrices,axis=1)
        M = X.shape[1]
        
        """
        CENTER data before PCA
        """
        xM = np.mean(X,axis=1)
        X = X - xM[:,np.newaxis]
        
        """
        PCA: U @ L @ U.T = eigenvalue decomposition of S=1/M * X @ X.T
        let U @ D @ V.T = X, then 1/n * X @ X.T = U @ 1/n*D**2 @ U.T
        """
        U,_,_ = np.linalg.svd(X,full_matrices=False)
        
        assert U.shape[0] == N, f'X.shape = {X.shape}; U.shape = {U.shape}'
        
        Z = U[:,:2].T @ X
        
        assert Z.shape == (2,M), f'X.shape = {X.shape}; Z.shape = {Z.shape}'
        
        """plotting"""
        plt.figure()
        self._scatterplots = []
        
        nof_matrices = len(matrices)
        if nof_matrices == 1:
            self._plot(Z)
        else:
            i = 0
            for k in range(nof_matrices):
                j = i + sizes[k]
                self._plot(Z[:,i:j])
                i = j
                
        if labels is not None:
            plt.legend(self._scatterplots,labels)
            
        plt.show()
            
    def _plot(self,Z):
        self._scatterplots.append(plt.scatter(Z[0,:],Z[1,:]))

if __name__ == '__main__':
    A = np.array([[2,2],[3,4],[9,3],[2,1],[7,3]]).T
    B = np.array([[3,3],[4,5],[6,1],[2,8]]).T
    C = np.array([[2,2,5],[3,4,3],[9,3,7],[2,1,3],[7,3,7]]).T
    D = np.array([[1,4,4],[2,9,3],[9,1,1],[4,8,5]]).T
    
    mp = MatrixPlotter()
    mp.plot(A,B,labels=['Alpha','Beta'])
    mp.plot(C,D,labels=['Gamma','Delta'])