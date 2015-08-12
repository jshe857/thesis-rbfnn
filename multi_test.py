#!/usr/bin/python2
import math
import time
import numpy as np

import sys
sys.path.append('EP/')
import PBP_net
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
np.random.seed(1)

#################### We load artificial data from an RBFNN ########################
n_dim = 13
n_nodes = 10
n_pts = 600
centers = np.random.rand(n_nodes,n_dim)

x_pts =  np.linspace(-50,50,num=num_pts)

#################### We load the boston housing dataset ###########################
data = np.loadtxt('boston_housing.txt')
X = data[ :, range(data.shape[ 1 ] - 1) ]
y = data[ :, data.shape[ 1 ] - 1 ]

#################### We load concrete dataset ######################################
#csv = np.genfromtxt ('concrete.csv', delimiter=",",skip_header=1)
#X = csv[ :, range(csv.shape[ 1 ] - 3) ]
#y = csv[ :, csv.shape[ 1 ] - 1 ]

##################### We load forestfires dataset #################################
# csv = np.genfromtxt ('forestfires.csv', delimiter=",",skip_header=1)

# ind = range(csv.shape[ 1 ] - 1)
# ind = [x for x in ind if (x != 2 and x != 3)]
# X = csv[ :,ind]
# y = csv[ :, csv.shape[ 1 ] - 1 ]

# for i in range(len(y)):
    # if y[i] > 0:
        # y[i] = np.log(y[i])

###################### We load the word music dataset ###############################
# csv = np.genfromtxt ('music.csv', delimiter=",",skip_header=1)
# X = csv[ :, range(csv.shape[ 1 ] - 2) ]
# y = csv[ :, csv.shape[ 1 ] - 1 ]


# We create the train and test sets with 90% and 10% of the data

print X.shape
print y.shape


permutation = np.random.choice(range(X.shape[ 0 ]),
    X.shape[ 0 ], replace = False)
size_train = np.round(X.shape[ 0 ] * 0.8)
index_train = permutation[ 0 : size_train ]
index_test = permutation[ size_train : ]

X_train = X[ index_train, : ]
y_train = y[ index_train ]
X_test = X[ index_test, : ]
y_test = y[ index_test ]

# We construct the network with one hidden layer with two-hidden layers
# with 50 neurons in each one and normalizing the training features to have
# zero mean and unit standard deviation in the trainig set.

n_hidden_units = 20
skip_len = 5
net = PBP_net.PBP_net(X_train[: 2], y_train[: 2],
    [n_hidden_units ], normalize = True, n_epochs = 1)

m, v, v_noise = net.predict(X_test)
rmse_test = np.sqrt(np.mean((y_test - m)**2))

plt.ion()
plt.show()

patch_test = mpatches.Patch(color='blue', label='rmse_test')
patch_train = mpatches.Patch(color='red', label='rmse_train')
plt.legend(handles=[patch_test,patch_train])

m, v, v_noise = net.predict(X_train)
rmse_train = np.sqrt(np.mean((y_train - m)**2))

plot = plt.plot(0,rmse_test,'bx',label='test')
plt.plot(0,rmse_train,'ro',label='train')
plt.draw()

run = 1

for j in range(10):
#We make predictions for the test set
    for i in range(0,len(X_train),skip_len):
        skip = min(i+skip_len,len(X_train)-1)
        net.re_train(X_train[i:i+skip],y_train[i:i+skip],1)
        m, v, v_noise = net.predict(X_test)
        rmse_test = np.sqrt(np.mean((y_test - m)**2))
        m, v, v_noise = net.predict(X_train)
        rmse_train = np.sqrt(np.mean((y_train - m)**2))
        plt.plot(run,rmse_test,'bx',label='test')
        plt.plot(run,rmse_train,'ro',label='train')
        plt.draw()
        run += 1

# We compute the test RMSE
m, v, v_noise = net.predict(X_test)
rmse = np.sqrt(np.mean((y_test - m)**2))

print
print 'rmse'
print rmse

# We compute the test log-likelihood

test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
    0.5 * (y_test - m)**2 /(v + v_noise))
print "test_log likelihood"
print test_ll

plt.savefig('result.png')
