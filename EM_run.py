#!/usr/bin/python2
import math
import time
import numpy as np
from sklearn import svm
from sklearn import cluster
from sklearn import linear_model
import sys
sys.path.append('EM/')
import EM_net
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


################# EM for RBFNN approach #############################################
def em_run(X_train,y_train,X_test,y_test,n_hidden_units,lam=0.1,eta=1,a=0.01):
    em = EM_net.EM_net(X_train,y_train, n_hidden_units,lam,eta,a)
    em.sgd(X_train,y_train,n_epochs=20)
    

    # line_test = plt.plot(n,result,label='Test Error')
    # line_train = plt.plot(n,result_train,label='Train Error')
    # plt.legend()
    # plt.show()

    # print '====================EM========================'
    rbf_sgd = em.sgd_predict(X_test)
    test_res = np.sqrt(np.mean((y_test - rbf_sgd)**2))
    # print 'test error'
    # print test_res

    rbf_sgd = em.sgd_predict(X_train)
    train_res = np.sqrt(np.mean((y_train - rbf_sgd)**2))
    # print 'train error'
    # print train_res




    result = svm.SVR().fit(X_train,y_train).predict(X_test) 
    svr_res = np.sqrt(np.mean((y_test - result)**2))

    # print '==================SVR========================'
    # print svr_res

    result = linear_model.LinearRegression().fit(X_train,y_train) \
            .predict(X_test)

    lm_res = np.sqrt(np.mean((y_test - result)**2))
   
    # print '==================Linear Model========================'
    # print lm_res

    return {'train':train_res,'test':test_res,'svr':svr_res,'lm':lm_res} 
