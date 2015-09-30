#!/usr/bin/python2
import math
import time
import numpy as np
import Data
import sys
sys.path.append('EM/')
sys.path.append('EP/')
import EM_net
import EP_net

def compute_result(y,result,ccc):
    if (ccc):
        e1 = np.mean(y)
        e2 = np.mean(result)
        a = y - e1
        b = result - e2 
        res = 2*np.mean(a*b)/(np.var(y) + np.var(result) + (e1 - e2)**2)
    else:
        res = np.sqrt(np.mean((y - result)**2))
    return res
def ep_run(X_train,y_train,X_test,y_test,n_hidden_units,lam=0.1,var_prior=1,ccc=False):
    net = EP_net.EP_net(X_train, y_train,
        [n_hidden_units ],lam,var_prior)
    net.train(X_train,y_train,10)
    m, v, v_noise = net.predict(X_test)
    test_res = compute_result(y_test,m,ccc)
    return test_res

def em_run(X_train,y_train,X_test,y_test,n_hidden_units,lam=0.1,eta=1,a=0.01,ccc=False):
    em = EM_net.EM_net(X_train,y_train, n_hidden_units,lam,eta,a)
    em.sgd(X_train,y_train,n_epochs=20)
    output = em.sgd_predict(X_test)
    test_res = compute_result(y_test,output,ccc)
    return test_res

def hyper_search(X,y,X_dev=None,y_dev=None,ccc=False):
    if X_dev==None:
	    dataset = Data.partition(X,y)
	    X_train = dataset['X_train']
	    y_train = dataset['y_train']
	    X_dev = dataset['X_dev']
	    y_dev = dataset['y_dev']
    else:
	    X_train = X
	    y_train = y
# Find Optimal Hyperparameter Setting
    lam_arr = [0.01,0.05,0.1,1,10]
    a_arr = [0,0.001,0.01,0.1,0.5,1]
    eta_arr = [0.01,0.1,0.3,0.5,0.7,0.9,1,1.1]
    n_arr = [10,20,30,50]
    var_prior_arr = [0.1,0.5,0.7,1,1.1,1.5,2]
    
    
    lam_arr = [0.009,0.01,0.02]
    a_arr = [0,0.01]
    eta_arr = [0.0001,0.001]
    n_arr = [50,150]
    var_prior_arr = [0.1,3]
   
    best_ep = -1e6
    best_em = -1e6
    params = {}
    for lam in lam_arr:
        for n in n_arr:
###############################FOR EP################################################
            for var_prior in var_prior_arr:
                err = ep_run(X_train,y_train,X_dev,y_dev,n,lam=lam,var_prior=var_prior,ccc=ccc)

                if(ccc and err > best_ep )or \
                (not ccc and err < best_ep):
                   best_ep = err
                   params['lam'] = lam
                   params['n'] = n
                   params['var_prior'] = var_prior
###############################FOR EM################################################
            for eta in eta_arr:
                for a in a_arr:
                    err = em_run(X_train,y_train,X_dev,y_dev,n,
                           lam=lam,eta=eta,a=a,ccc=ccc)
                    if(ccc and err > best_em )or \
                    (not ccc and err < best_em):
                       best_em = err
                       params['lam_em'] = lam
                       params['n_em'] = n
                       params['eta'] = eta
                       params['a'] = a
            print params
	    print "best EP error: " + str(best_ep)
	    print "best EM error: " + str(best_em)
    print "best params"
    print params



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
# print 'Artificial Optimal HyperParameter'

#################### We load the boston housing dataset ###########################
#data = np.loadtxt('boston_housing.txt')
#X = data[ :, range(data.shape[ 1 ] - 1) ]
#y = data[ :, data.shape[ 1 ] - 1 ]
#print 'Boston Housing Optimal HyperParameter'
#################### We load concrete dataset ######################################
# csv = np.genfromtxt ('concrete.csv', delimiter=",",skip_header=1)
# X = csv[ :, range(csv.shape[ 1 ] - 3) ]
# y = csv[ :, csv.shape[ 1 ] - 1 ]
# print 'Concrete Optimal HyperParameter'
# hyper_search(X,y)
# y = csv[ :, csv.shape[ 1 ] - 2]
# hyper_search(X,y)
# ##################### We load forestfires dataset #################################
# csv = np.genfromtxt ('forestfires.csv', delimiter=",",skip_header=1)

# ind = range(csv.shape[ 1 ] - 1)
# ind = [x for x in ind if (x != 2 and x != 3)]
# X = csv[ :,ind]
# y = csv[ :, csv.shape[ 1 ] - 1 ]

# for i in range(len(y)):
    # if y[i] > 0:
        # y[i] = np.log(y[i])
# print 'Forestfires Optimal HyperParameter'
# hyper_search(X,y)

################ Artificial 1D dataset ##########################################
# num_train = 1000
# def generate_xy(rng,num,noise=True):
    # x_pts =  np.linspace(-rng,rng,num=num)
    # X = np.array([x_pts]).T
    # if (noise):
        # y = 3*np.cos(x_pts/9) +  2*np.sin(x_pts/15) + 0.1*np.random.randn(num)
    # else:
        # y = 3*np.cos(x_pts/9) +  2*np.sin(x_pts/15)
    # return(X,y)
# rng = 80
# X,y = generate_xy(rng,num_train)
# hyper_search(X,y)

###################### We load the word music dataset ###############################
# csv = np.genfromtxt ('music.csv', delimiter=",",skip_header=1)
# X = csv[ :, range(csv.shape[ 1 ] - 2) ]
# y = csv[ :, csv.shape[ 1 ] - 1 ]
# print 'World Music  Optimal HyperParameter'
# hyper_search(X,y)
# y = csv[ :, csv.shape[ 1 ] - 2 ]
# hyper_search(X,y)




