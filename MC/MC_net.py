import pymc3 as pm
import numpy as np
from scipy import optimize
import theano
import theano.tensor as T
class MC_net:
    def rbf_kernel(self,C,x):
         return np.exp(-T.sum((C-x)**2))
        

    def __init__(self,X_train,y_train,n_hidden,lam=1):
        n_train = y_train.shape[0]
        n_dim = X_train.shape[1]
        print X_train.shape
        with pm.Model() as rbfnn:
            C = pm.Normal('C',shape=(n_hidden,n_dim))
            beta = pm.Gamma('beta',10,10)
            w = pm.Normal('w',shape=(1,n_hidden))
            
            #component, updates = theano.scan(fn=lambda x: T.sum(C-x)**2,sequences=[X_train])
            y_out=[]
            for x in X_train:
               y_out.append(T.sum(w*T.exp(-T.sum(C-x)**2)))
            
            y = pm.Normal('y',y_out,beta,observed=y_train)
            start = pm.find_MAP()
            print start

