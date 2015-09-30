#!/usr/bin/python2
import math
import time
import numpy as np
import sys
sys.path.append('EP/')
import EP_net
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def ep_run(X_train,y_train,X_test,y_test,n_hidden_units,lam=0.1,var_prior=1,ccc=False):
    ################### Construct RBFNN #################################################
    # print 'lam: ' + str(lam)
    # print 'var_prior: ' + str(var_prior)
    net = EP_net.EP_net(X_train, y_train,
        [n_hidden_units ],lam,var_prior)
    # We compute the test RMSE
    net.train(X_train,y_train,10)
    m, v, v_noise = net.predict(X_test)
    train_res = compute_result(y_test,m,ccc)
    # plt.plot(n,result)
    # print '====================EP========================'
    # print 'test error'
    # print rmse

    m, v, v_noise = net.predict(X_train)
    train_res = compute_result(y_train,m,ccc)
    #print 'train error'
    #print rmse

    # We compute the test log-likelihood

    # test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
        # 0.5 * (y_test - m)**2 /(v + v_noise))
    # print "test_log likelihood"
    # print test_ll
    return {'test':test_res,'train':train_res}
