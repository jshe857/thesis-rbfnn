import pymc3 as pm
from pymc3 import traceplot
from pymc3 import summary
import numpy as np
from scipy import optimize
import theano
import theano.tensor as T
import scipy
import cPickle as pickle
class MC_net:
    def rbf_kernel(self,C,x):
         return np.exp(-T.sum((C-x)**2))
        

    def __init__(self,X_train,y_train,n_hidden,lam=1):
        n_train = y_train.shape[0]
        n_dim = X_train.shape[1]
        print X_train.shape
        with pm.Model() as rbfnn:
            C = pm.Normal('C',mu=0,sd=10,shape=(n_hidden,n_dim))
            beta = pm.Gamma('beta',1,1)
            w = pm.Normal('w',mu=0,sd=10,shape=(1,n_hidden))
            
            #component, updates = theano.scan(fn=lambda x: T.sum(C-x)**2,sequences=[X_train])
            y_out=[]
            for x in X_train:
                #rbf_out =  T.exp(-lam*T.sum((C-x)**2,axis=1)) 
                #1d speed up
                rbf_out =  T.exp(-lam*(C-x)**2) 
                y_out.append(T.dot(w,rbf_out))
            
            y = pm.Normal('y',y_out,beta,observed=y_train)
            
            start = pm.find_MAP(fmin=scipy.optimize.fmin_l_bfgs_b)
            print start
            step = pm.NUTS(scaling=start)
            trace = pm.sample(2000, step, progressbar=False)
            step = pm.NUTS(scaling=trace[-1])
            trace = pm.sample(40000,step,start=trace[-1])
            traceplot(trace)        

            with open("tracedata.txt","wb") as output:
                pickle.dump(trace,output,pickle.HIGHEST_PROTOCOL)

            print summary(trace, vars=['C', 'w','beta'])
            mcmc = open("mcmc.txt","w")
            mcmc.write(summary(trace, vars=['C', 'w','beta']))
            mcmc.close()
