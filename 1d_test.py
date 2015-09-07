#!/usr/bin/python2
import math
import numpy as np
import sys
sys.path.append('MC')
import MC_net
# import matplotlib
# matplotlib.use('pgf')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
np.random.seed(1)
# We create the train and test sets with 90% and 10% of the data
#Generate artificial data
num_train = 50
num_test = 25
def generate_xy(rng,num,noise=True):
    x_pts =  np.linspace(-rng,rng,num=num)
    X = np.array([x_pts]).T
    if (noise):
        y = 3*np.cos(x_pts/9) +  2*np.sin(x_pts/15) + 0.1*np.random.randn(num)
    else:
        y = 3*np.cos(x_pts/9) +  2*np.sin(x_pts/15)
    return(X,y)
rng = 80
X,y = generate_xy(rng,num_train)
x_true,y_true = generate_xy(rng,200,noise=False)
print X.shape
print y.shape
permutation = np.random.choice(range(X.shape[ 0 ]),
    X.shape[ 0 ], replace = False)
size_train = num_train
index_train = permutation[ 0 : size_train ]
index_test = permutation[ size_train : ]

X_train = X[ index_train, : ]
y_train = y[ index_train ]
X_test,y_test = generate_xy(rng,num_test)

MC_net.MC_net(X_train,y_train,3,lam=1)

