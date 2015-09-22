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
np.random.seed(1)

def read_arff(glob_pattern,skip):
    dataset = [] 
    for files in glob.glob(glob_pattern):
        data =arff.load(open(files))
        dataset.append([row[skip:] for row in data['data']])
    return np.concatenate(dataset)

############################# Load and Process AVEC rows #############################
def read_avec(search_pattern):
    features = 'Data/avec/features_audio/'
    valence = 'Data/avec/ratings_gold_standard/valence/'
    arousal = 'Data/avec/ratings_gold_standard/arousal/'

    X = read_arff(features+search_pattern,1)
    y_valence = np.squeeze(read_arff(valence+search_pattern,2))
    y_arousal = np.squeeze(read_arff(arousal+search_pattern,2))
    return (X,y_valence,y_arousal)



(X_dev,y_dev1,y_dev2) = read_avec('dev_*')
(X,y_1,y_2) = read_avec('train_*')

permutation = np.random.choice(range(X.shape[ 0 ]),
    X.shape[ 0 ], replace = False)

size_train = np.round(X.shape[ 0 ] * 0.8)
index_train = permutation[ 0 : size_train ]
index_test = permutation[ size_train : ]
X_train = X[index_train] 
y_train1 = y_1[index_train]
y_train2 = y_2[index_train]

X_test = X[index_test] 
y_test1 = y_1[index_test]
y_test2 = y_2[index_test]

hyperparameter.hyper_search(X_train,y_train1,X_dev=X_dev,y_dev=y_dev1)
hyperparameter.hyper_search(X_train,y_train2,X_dev=X_dev,y_dev=y_dev2)

################### Construct RBFNN #################################################
# lam = 0.08
# var_prior = 1.0 
# n_hidden_units = 100
# print 'lam: ' + str(lam)
# print 'var_prior: ' + str(var_prior)
# print 'n_hidden_units: ' + str(n_hidden_units)

# # skip_len = 500
# net = EP_net.EP_net(X_train, y_train,
    # [n_hidden_units ],lam,var_prior)

# print
# # We compute the test RMSE
# net.train(X_train,y_train,40)
# m, v, v_noise = net.predict(X_test)
# rmse = np.sqrt(np.mean((y_test - m)**2))

# ######################### EP for RBFNN approach #############################################
# print '====================EP========================'
# print 'test error'
# print rmse

# m, v, v_noise = net.predict(X_train)
# rmse = np.sqrt(np.mean((y_train - m)**2))
# print 'train error'
# print rmse

# ######################### EM for RBFNN approach #############################################
# em = EM_net.EM_net(X_train,y_train, n_hidden_units,lam)

# skip_len=10
# for i in range(0,len(X_train),skip_len):
    # skip = min(i+skip_len,len(X_train)-1)
    # em.sgd(X_train[i:skip],y_train[i:skip])
    # #print 'weights'
    # #print em.get_sgd_weights()
# rbf_sgd = em.sgd_predict(X_test)
# rmse = np.sqrt(np.mean((y_test - rbf_sgd)**2))

# print '====================EM========================'
# print 'test error'
# print rmse

# rbf_sgd = em.sgd_predict(X_train)
# rmse = np.sqrt(np.mean((y_train - rbf_sgd)**2))
# print 'train error'
# print rmse


# result = svm.SVR().fit(X_train,y_train).predict(X_test) 
# svr_res = np.sqrt(np.mean((y_test - result)**2))
# print '==================SVR========================'
# print svr_res
