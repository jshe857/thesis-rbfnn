#!/usr/bin/python2
import math
import time
import numpy as np
from sklearn import svm
from sklearn import cluster
import sys
sys.path.append('EM/')
import EM_net
sys.path.append('EP/')
import EP_net
import Data
import matplotlib
import seaborn as sns
# matplotlib.use('pgf')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import theano
sns.set_context('poster')

# theano.config.profile = True

def run_test(X,y,a,lam_em,lam,eta,n,n_em,var_prior):
    ################### Construct RBFNN #################################################
    print 'lam: ' + str(lam)
    print 'var_prior: ' + str(var_prior)
    print 'n_hidden_units: ' + str(n)

    dataset = Data.partition(X,y)
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    net = EP_net.EP_net(X_train, y_train,
        [n ],lam,var_prior)

    # We compute the test RMSE
    net.train(X_train,y_train,40)
    m, v, v_noise = net.predict(X_test)
    print
    print '====================EP========================'
    print 'test error'
    rmse = np.sqrt(np.mean((y_test - m)**2))
    print rmse

    ################# EM for RBFNN approach #############################################
    em = EM_net.EM_net(X_train,y_train, n_em,lam_em,eta,a)

    em.sgd(X_train,y_train)
    rbf_sgd = em.sgd_predict(X_test)
    rmse = np.sqrt(np.mean((y_test - rbf_sgd)**2))

    print '====================EM========================'
    print 'test error'
    print rmse



    # We compute the test log-likelihood

    # test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
        # 0.5 * (y_test - m)**2 /(v + v_noise))
    # print "test_log likelihood"
    # print test_ll

    result = svm.SVR().fit(X_train,y_train).predict(X_test) 
    svr_res = np.sqrt(np.mean((y_test - result)**2))
    print '==================SVR========================'
    print svr_res
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
# #################### We load the boston housing dataset ###########################
# print "Boston"
# data = np.loadtxt('boston_housing.txt')
# X = data[ :, range(data.shape[ 1 ] - 1) ]
# y = data[ :, data.shape[ 1 ] - 1 ]
# var_prior = 1.0 
# n = 40
# n_em = 50
# lam_em = 0.01
# lam = 0.05
# eta = 0.0212
# a=0

# run_test(X,y,a,lam_em,lam,eta,n,n_em,var_prior)
#################### We load concrete dataset 2######################################
# print "Concrete"
# csv = np.genfromtxt ('concrete.csv', delimiter=",",skip_header=0)
# X = csv[ :, range(csv.shape[ 1 ] - 1) ]
# y = csv[ :, csv.shape[ 1 ] - 1]
# a=0
# lam_em=0.01
# lam = 0.1
# eta=0.3
# n=50
# n_em=20
# var_prior=0.1

# run_test(X,y,a,lam_em,lam,eta,n,n_em,var_prior)
##################### We load forestfires dataset #################################
# print "Forest"
# csv = np.genfromtxt ('forestfires.csv', delimiter=",",skip_header=1)

# ind = range(csv.shape[ 1 ] - 1)
# ind = [x for x in ind if (x != 2 and x != 3)]
# X = csv[ :,ind]
# y = csv[ :, csv.shape[ 1 ] - 1 ]

# for i in range(len(y)):
    # if y[i] > 0:
        # y[i] = np.log(y[i])
# a=0
# lam_em=0.1
# lam = 0.05
# eta=0.1
# n_em = 10
# n =20
# var_prior=0.5

# run_test(X,y,a,lam_em,lam,eta,n,n_em,var_prior)

# ###################### We load the word music dataset ###############################
# print "Music 1"
csv = np.genfromtxt ('music.csv', delimiter=",",skip_header=1)
X = csv[ :, range(csv.shape[ 1 ] - 2) ]
# y = csv[ :, csv.shape[ 1 ] - 1 ]
# a=0
# lam_em=0.01
# lam=0.05
# eta =0.1
# n=10
# n_em=20
# var_prior=0.5

# run_test(X,y,a,lam_em,lam,eta,n,n_em,var_prior)

y = csv[ :, csv.shape[ 1 ] - 2 ]
a=0
lam_em=0.05
lam=0.01
eta =0.3
n=20
n_em=50
var_prior=0.1


print "Music 2"
run_test(X,y,a,lam_em,lam,eta,n,n_em,var_prior)
