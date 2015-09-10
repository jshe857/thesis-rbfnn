import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn
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
            C = pm.Normal('C',mu=0,sd=10,shape=(n_hidden))
            #beta = pm.Gamma('beta',1,1)
            w = pm.Normal('w',mu=0,sd=10,shape=(n_hidden+1))
            
            #component, updates = theano.scan(fn=lambda x: T.sum(C-x)**2,sequences=[X_train])
            y_out=[]
            for x in X_train:
                #rbf_out =  T.exp(-lam*T.sum((C-x)**2,axis=1)) 
                #1d speed up
                rbf_out =  T.exp(-lam*(C-x)**2)
                #rbf_out = theano.printing.Print(rbf_out)                 
                rbf_out_biased = \
                        T.concatenate([ rbf_out, T.alloc(1,1) ], 0)
                y_out.append(T.dot(w,rbf_out_biased))
            
            y = pm.Normal('y',mu=y_out,sd=0.01,observed=y_train)
            
            start = pm.find_MAP(fmin=scipy.optimize.fmin_l_bfgs_b)
            print start
            step = pm.NUTS(scaling=start)
            trace = pm.sample(2000, step, progressbar=False)
            step = pm.NUTS(scaling=trace[-1])
            trace = pm.sample(20000,step,start=trace[-1])
            

            print summary(trace, vars=['C', 'w'])

            vars = trace.varnames   
            for i, v in enumerate(vars):
                for d in trace.get_values(v, combine=False, squeeze=False):
                    d=np.squeeze(d)
                    with open(str(v)+".txt","w+") as thefile:
                        for item in d:
                            print>>thefile, item

            traceplot(trace)
            plt.show()
