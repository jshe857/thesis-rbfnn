#!/usr/bin/python2
import math
import time
import numpy as np
import EP_run
import EM_run
import sys
sys.path.append('MC')
import MC_net

import Data
from theano import config
#################### We load artificial data from an RBFNN ########################
# n_dim = 10
# n_nodes = 10
# n_pts = 100
# c = np.random.rand(n_nodes,n_dim)*2
# w = np.random.rand(n_nodes,1)*3
# # generate random inputs with gaussian noise
# X =  (np.random.rand(n_pts,n_dim) - 0.5)*4 + np.random.randn(n_pts,n_dim)
# y = []
# for x_in in X:
      # #rbfs = np.exp(-0.1*np.sum((x_in - c)**2,axis=1))
      # #y.append(np.sum(w*rbfs))
      # sins = np.sin(2*x_in)
      # y.append(np.sum(2*sins))
# y = np.array(y + 1*np.random.randn(n_pts))
# eta = 1.1
# a = 0.1
# n_hidden_units = 50
#################### We load the boston housing dataset ###########################
data = np.loadtxt('boston_housing.txt')
X = data[ :, range(data.shape[ 1 ] - 1) ]
y = data[ :, data.shape[ 1 ] - 1 ]
n = 50
a = 0
eta = 0.0212
lam_em = 0.01
lam = 0.05
var_prior = 0.76
#################### We load concrete dataset ######################################
#csv = np.genfromtxt ('concrete.csv', delimiter=",",skip_header=1)
#X = csv[ :, range(csv.shape[ 1 ] - 3) ]
#y = csv[ :, csv.shape[ 1 ] - 1 ]

##################### We load forestfires dataset #################################
#csv = np.genfromtxt ('forestfires.csv', delimiter=",",skip_header=1)

#ind = range(csv.shape[ 1 ] - 1)
#ind = [x for x in ind if (x != 2 and x != 3)]
#X = csv[ :,ind]
#y = csv[ :, csv.shape[ 1 ] - 1 ]

#for i in range(len(y)):
    #if y[i] > 0:
        #y[i] = np.log(y[i])

###################### We load the word music dataset ###############################
# csv = np.genfromtxt ('music.csv', delimiter=",",skip_header=1)
# X = csv[ :, range(csv.shape[ 1 ] - 2) ]
# y = csv[ :, csv.shape[ 1 ] - 1 ]

#config.profile=True

################### We load power dataset ######################################
csv = np.genfromtxt ('power.csv', delimiter=",",skip_header=1)
X = csv[ 1:100, range(csv.shape[ 1 ] - 1) ]
y = csv[ 1:100, csv.shape[ 1 ] - 1 ]




dataset = Data.partition(X,y)
X_train = np.append(dataset['X_train'],dataset['X_dev'],axis=0)
y_train = np.append(dataset['y_train'],dataset['y_dev'],axis=0)
X_test = dataset['X_test']
y_test = dataset['y_test']
result = {'ep_train':0,'ep_test':0,
        'em_train':0,'em_test':0,'svr':0}

for s in range(9):
# Find Optimal Hyperparameter Setting
    np.random.seed(s)
    r = EP_run.ep_run(X_train,y_train,X_test,y_test,n,lam=lam,var_prior=var_prior) 
    result['ep_train'] += r['train']
    result['ep_test'] += r['test']
    r = EM_run.em_run(X_train,y_train,X_test,y_test,n,lam=lam_em,eta=eta,a=a) 
    result['em_train'] += r['train']
    result['em_test'] += r['test']
    result['svr'] += r['svr']
result['ep_train'] /= 9
result['em_train'] /= 9
result['ep_test'] /= 9
result['em_test'] /= 9
result['svr'] /= 9
# Development Set
print 'for size '+str(n)+':'
print result

# Perform K-fold cross validation




