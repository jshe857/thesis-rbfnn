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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
np.random.seed(1)

#################### We load artificial data from an RBFNN ########################
n_dim = 10
n_nodes = 10
n_pts = 1000
#c = np.random.rand(n_nodes,n_dim)*2
#w = np.random.rand(n_nodes,1)*3
#generate random inputs with gaussian noise
#X =  (np.random.rand(n_pts,n_dim) - 0.5)*4 + np.random.randn(n_pts,n_dim)
#y = []
#for x_in in X:
     ##rbfs = np.exp(-0.1*np.sum((x_in - c)**2,axis=1))
     ##y.append(np.sum(w*rbfs))
     #sins = np.sin(2*x_in)
     #y.append(np.sum(2*sins))
#y = np.array(y + 1*np.random.randn(n_pts))
#################### We load the boston housing dataset ###########################
data = np.loadtxt('boston_housing.txt')
X = data[ :, range(data.shape[ 1 ] - 1) ]
y = data[ :, data.shape[ 1 ] - 1 ]

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
#csv = np.genfromtxt ('music.csv', delimiter=",",skip_header=1)
#X = csv[ :, range(csv.shape[ 1 ] - 2) ]
#y = csv[ :, csv.shape[ 1 ] - 1 ]


# We create the train and test sets with 80% and 20% of the data
print 'X'
print X.shape
print 'y'
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


################### Construct RBFNN #################################################
lam = 0.01
n_hidden_units = 10
# skip_len = 500
net = EP_net.EP_net(X_train, y_train,
    [n_hidden_units ],lam)

# m, v, v_noise = net.predict(X_test)
# rmse_test = np.sqrt(np.mean((y_test - m)**2))

# m, v, v_noise = net.predict(X_train)
# rmse_train = np.sqrt(np.mean((y_train - m)**2))

################## Start plot #######################################################
#plt.ion()
#plt.show()
#patch_test = mpatches.Patch(color='blue', label='rmse_test')
#patch_train = mpatches.Patch(color='red', label='rmse_train')
#plt.legend(handles=[patch_test,patch_train])
#plot = plt.plot(0,rmse_test,'bx',label='test')
#plt.plot(0,rmse_train,'ro',label='train')
#plt.draw()

################# Plot RMSE as we go ################################################
#run = 1
#for j in range(40):
    #print '============iteration================'
#We make predictions for the test set
    #for i in range(0,len(X_train),skip_len):
        #skip = min(i+skip_len,len(X_train)-1)
        #net.train(X_train[i:skip],y_train[i:skip],1)
        
        #m, v, v_noise = net.predict(X_test)
        #rmse_test = np.sqrt(np.mean((y_test - m)**2))
        #m, v, v_noise = net.predict(X_train)
        #rmse_train = np.sqrt(np.mean((y_train - m)**2))
        #plt.plot(run,rmse_test,'bx',label='test')
        #plt.plot(run,rmse_train,'ro',label='train')
        #plt.draw()
        #run += 1
# We compute the test RMSE
net.train(X_train,y_train,40)
m, v, v_noise = net.predict(X_test)
rmse = np.sqrt(np.mean((y_test - m)**2))
print 
print '====================EP========================'
print 'test error'
print rmse

m, v, v_noise = net.predict(X_train)
rmse = np.sqrt(np.mean((y_train - m)**2))
print 'train error'
print rmse
################# EM for RBFNN approach #############################################
em = EM_net.EM_net(X_train,y_train, n_hidden_units,lam)

#em.pseudo_inverse(X_train,y_train)
#rbf_pseudo = em.pseudo_predict(X_test)
#rmse = np.sqrt(np.mean((y_test - rbf_pseudo)**2))
#print 'EM-pseudo'
#print rmse


skip_len=10
for i in range(0,len(X_train),skip_len):
    skip = min(i+skip_len,len(X_train)-1)
    em.sgd(X_train[i:skip],y_train[i:skip])
    #print 'weights'
    #print em.get_sgd_weights()
rbf_sgd = em.sgd_predict(X_test)
rmse = np.sqrt(np.mean((y_test - rbf_sgd)**2))

print '====================EM========================'
print 'test error'
print rmse

rbf_sgd = em.sgd_predict(X_train)
rmse = np.sqrt(np.mean((y_train - rbf_sgd)**2))
print 'train error'
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
