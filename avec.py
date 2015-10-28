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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import hyperparameter
import arff
import glob
import theano
theano.config.profile = True
# np.random.seed(1)

TIME_STEP = 0.04
VALENCE_SKIP = int(2/TIME_STEP)
AROUSAL_SKIP = int(4/TIME_STEP)
def read_arff(glob_pattern,time_delay,read_X=False):
    dataset = []
    skip_header = 2
    if read_X:
            skip_header=1

    for files in sorted(glob.glob(glob_pattern)):
        data =arff.load(open(files))['data']
	if read_X:
		data = data [:-time_delay]
	else:
		data = data[time_delay:]
        dataset.append([row[skip_header:] for row in data])
    # data = np.concatenate(dataset)
    # if read_X:
	# data = data [:-time_delay]
    # else:
	# data = data[time_delay:]
    # return data
    return np.concatenate(dataset)
def read_csv(glob_pattern,time_delay):
    dataset = []
    skip_col = 1
    for files in sorted(glob.glob(glob_pattern)):
        data = np.genfromtxt(files,delimiter=";",skip_header=1)
        dataset.append(data[:,1:])
    return np.concatenate(dataset)
# def read_arff(glob_pattern,time_delay,read_X=False):
    # dataset = []
    # skip_header = 2
    # if read_X:
	    # skip_header=1

    # for files in glob.glob(glob_pattern):
        # data =arff.load(open(files))['data']
        # dataset.append([row[skip_header:] for row in data])
    # if read_X:
        # return np.concatenate(dataset)[:-time_delay]
    # return np.concatenate(dataset)[time_delay:]
############################# Load and Process AVEC rows #############################
def read_avec(search_pattern):
    features = 'Data/avec/features_audio/'
    valence = 'Data/avec/ratings_gold_standard/valence/'
    arousal = 'Data/avec/ratings_gold_standard/arousal/'

    X_valence = read_arff(features+search_pattern,VALENCE_SKIP,read_X=True)
    X_arousal = read_arff(features+search_pattern,AROUSAL_SKIP,read_X=True)
    y_valence = np.squeeze(read_arff(valence+search_pattern,VALENCE_SKIP))
    y_arousal = np.squeeze(read_arff(arousal+search_pattern,AROUSAL_SKIP))
    return (X_valence,X_arousal,y_valence,y_arousal)
def read_individual_avec(search_pattern):
    valence = 'Data/avec/ratings_individual/valence/'
    arousal = 'Data/avec/ratings_individual/arousal/'
    y_valence = np.squeeze(read_csv(valence+search_pattern,VALENCE_SKIP))
    y_arousal = np.squeeze(read_csv(arousal+search_pattern,AROUSAL_SKIP))
    return (y_valence,y_arousal)

def compute_ccc(y,result):
    e1 = np.mean(y)
    e2 = np.mean(result)
    a = y - e1
    b = result - e2 
    res = 2*np.mean(a*b)/(np.var(y) + np.var(result) + (e1 - e2)**2)
    return res

# ######################### EP for RBFNN approach #############################################
(X_dev1,X_dev2,y_dev1,y_dev2) = read_avec('dev_*')
(X_train1,X_train2,y_train1,y_train2) = read_avec('train_*')
# hyperparameter.hyper_search(X_train[:-VALENCE_SKIP],y_train1,ccc=True)
# hyperparameter.hyper_search(X_train[:-AROUSAL_SKIP],y_train2,ccc=True)
#VALENCE

lam = 0.02
var_prior = 0.1 
n_hidden_units = 150
print 'lam: ' + str(lam)
print 'var_prior: ' + str(var_prior)
print 'n_hidden_units: ' + str(n_hidden_units)

net = EP_net.EP_net(X_train1, y_train1,
    [n_hidden_units ],lam,var_prior)
net.train(X_train1,y_train1,2)




# m, v, v_noise = net.predict(X_dev1)
# print "VALENCE" 
# for (mean,var) in zip(m,v):
    # print [mean,var]
# ccc = compute_ccc(y_dev1,m)
# print '====================EP========================'
# print 'Valence CCC:'
# print ccc 


#AROUSAL
# lam = 0.02
# var_prior = 3 
# n_hidden_units = 100
# # n_hidden_units = 50
# print 'lam: ' + str(lam)
# print 'var_prior: ' + str(var_prior)
# print 'n_hidden_units: ' + str(n_hidden_units)
# for i in range(1,20,2):
    # np.random.seed(i)
    # permutation = np.random.choice(range(X_train2.shape[ 0 ]),
    # X_train2.shape[ 0 ], replace = False)
    # X_train2 = X_train2[permutation, :]
    # y_train2 = y_train2[permutation]
    # net = EP_net.EP_net(X_train2, y_train2,
        # [n_hidden_units ],lam,var_prior)
    # net.train(X_train2,y_train2,1)

    # m, v, v_noise = net.predict(X_dev2)
    # print "AROUSAL" 
    # for (mean,var) in zip(m,v):
        # print [mean,var]
    # ccc = compute_ccc(y_dev2,m)
    # print 'Arousal CCC:'
    # print ccc

# ######################### EM for RBFNN approach #############################################

# n=150
# lam = 0.009
# eta = 0.001
# a = 0
# em = EM_net.EM_net(X_train1,y_train1, n,lam,eta,a)

# em.sgd(X_train1,y_train1)
# rbf_sgd = em.sgd_predict(X_dev1)
# ccc = compute_ccc(y_dev1,rbf_sgd)

# print '====================EM========================'
# print 'EM Valence:'
# print ccc


# n=150
# lam = 0.009
# eta = 0.0001
# a = 0
# em = EM_net.EM_net(X_train2,y_train2, n,lam,eta,a)

# em.sgd(X_train2,y_train2)
# rbf_sgd = em.sgd_predict(X_dev2)
# ccc = compute_ccc(y_dev2,rbf_sgd)

# print 'EM Arousal:'
# print ccc

# result = svm.SVR().fit(X_train,y_train).predict(X_test) 
# svr_res = np.sqrt(np.mean((y_test - result)**2))
# print '==================SVR========================'
# print svr_res
