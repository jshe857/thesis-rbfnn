#!/usr/bin/python2
import math
import time
import numpy as np
import EP_run
import EM_run
import Data
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




dataset = Data.partition(X,y)
X_train = dataset['X_train']
y_train = dataset['y_train']
X_test = dataset['X_test']
y_test = dataset['y_test']
# Find Optimal Hyperparameter Setting
print '==========================EP========================'
print EP_run.ep_run(X_train,y_train,X_test,y_test,n,lam=lam,var_prior=var_prior) 
print '==========================EM========================'

print EM_run.em_run(X_train,y_train,X_test,y_test,n,lam=lam_em,eta=eta,a=a) 
# Development Set


# Perform K-fold cross validation




