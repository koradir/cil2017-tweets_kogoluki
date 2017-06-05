#!/usr/bin/env python3
from scipy.sparse import coo_matrix
import numpy as np
import numpy.random as nprnd
import pickle
from statusbar import status_update

def f_func(n,nmax,alpha):
    return min([1,(n/nmax)**alpha])

def main():
    print("loading cooccurrence matrix")
    with open('cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print(f"{cooc.nnz} nonzero entries")

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 300
    X = nprnd.normal(size=(cooc.shape[0], embedding_dim))
    Y = nprnd.normal(size=(cooc.shape[1], embedding_dim))
    
    print(X.shape)
    
    step = 0.001
    alpha = 3 / 4

    epochs = 50
    
    pmax = len(cooc.row)-1
    print(pmax)

    for epoch in range(epochs):
        p = 0
#        label=f"epoch {epoch}"
        print(f'epoch {epoch + 1} of {epochs}')
#        status_update(p,pmax,label=label)
        for i, j, n in zip(cooc.row, cooc.col, cooc.data):
            mij = np.log(n)
            fn = f_func(n,nmax,alpha)
            XTYij = X[i,:].T @ Y[j,:]  # note: vectors stored in rows, not columns
            
            scale = -2 * fn * (mij - XTYij)
            
            gradXi = scale * Y[j,:]
            gradYj = scale * X[i,:]
            
            X[i,:] -= step * gradXi
            Y[j,:] -= step * gradYj
            
#            p += 1
#            status_update(p,pmax,label=label)

    np.save(f'embeddingsX_K{embedding_dim}_step{step}_epochs{epochs}', X)
    #np.save('embeddingsY', Y)


if __name__ == '__main__':
    main()
