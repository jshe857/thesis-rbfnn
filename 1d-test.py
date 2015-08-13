#!/usr/bin/python2
import math

import numpy as np

import sys
sys.path.append('EP/')
import PBP_net
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
np.random.seed(1)
# We create the train and test sets with 90% and 10% of the data
#Generate artificial data
num_pts = 800
x_pts =  np.linspace(-50,50,num=num_pts)
X = np.array([x_pts]).T
y = 10*np.exp(-0.05*np.absolute(x_pts - 60)) + 10*np.exp(-0.05*np.absolute(x_pts)) +  0*x_pts + 1*np.random.randn(num_pts)
y = 2*np.cos(x_pts/5) +  2*np.sin(x_pts/30) + 0.1*np.random.randn(num_pts)


print X.shape
print y.shape
permutation = np.random.choice(range(X.shape[ 0 ]),
    X.shape[ 0 ], replace = False)
size_train = np.round(X.shape[ 0 ] * 0.9)
index_train = permutation[ 0 : size_train ]
index_test = permutation[ size_train : ]

X_train = X[ index_train, : ]
y_train = y[ index_train ]
X_test = X[ index_test, : ]
y_test = y[ index_test ]

# We construct the network with one hidden layer with two-hidden layers
# with 50 neurons in each one and normalizing the training features to have
# zero mean and unit standard deviation in the trainig set.

skip_len = num_pts/10
n_hidden_units = 60
net = PBP_net.PBP_net(X_train, y_train,
    [n_hidden_units])

m, v, v_noise = net.predict(X_test)
plt.plot(X_test,y_test,'ro',X_test,m,'bx')
plt.show()

net.train(X_train,y_train,20)
m, v, v_noise = net.predict(X_test)
plt.plot(X_test,y_test,'ro',X_test,m,'bx')
plt.show()
# We make predictions for the test set
#for i in range(0,len(X_train),skip_len):
    #skip = min(i+skip_len,len(X_train)-1)
    #net.train(X_train[i:skip],y_train[i:skip],1)
    #m, v, v_noise = net.predict(X_test)
    #plt.plot(X_test,y_test,'ro',X_test,m,'bx')
    #red_patch = mpatches.Patch(color='red', label='Test data')
    #blue_patch = mpatches.Patch(color='blue', label='Prediction')
    #plt.legend(handles=[red_patch,blue_patch])
    #plt.show()
# We compute the test RMSE

rmse = np.sqrt(np.mean((y_test - m)**2))
print
print 'rmse'
print rmse

# We compute the test log-likelihood

test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
    0.5 * (y_test - m)**2 / (v + v_noise))
print "test_log likelihood"
print test_ll
