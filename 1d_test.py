#!/usr/bin/python2
import math
import numpy as np
import sys
sys.path.append('MC')
sys.path.append('EP')
# import MC_net
import EP_net
# import matplotlib
# matplotlib.use('pgf')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
np.random.seed(1)
# We create the train and test sets with 90% and 10% of the data
#Generate artificial data
num_train = 20
num_test = 500
def generate_xy(rng,num,noise=True):
    x_pts =  np.linspace(-rng,rng,num=num)
    X = np.array([x_pts]).T
    if (noise):
        #y = 3*np.cos(x_pts/9) +  2*np.sin(x_pts/15) + 0.1*np.random.randn(num)
        y = 2*np.exp(-10*(x_pts - 0.1)**2) + 0.1*np.random.randn(num) 
    else:
        #y = 3*np.cos(x_pts/9) +  2*np.sin(x_pts/15)
        y = 2*np.exp(-10*(x_pts - 0.1)**2)
    return(X,y)
#rng = 50
rng = 2
X,y = generate_xy(rng,num_train)
x_true,y_true = generate_xy(rng,2,noise=False)
print X.shape
print y.shape

################### We load concrete dataset ######################################
#csv = np.genfromtxt ('concrete.csv', delimiter=",",skip_header=1)
#X = csv[ :, range(csv.shape[ 1 ] - 3) ]
#y = csv[ :, csv.shape[ 1 ] - 1 ]

################### We load power dataset ######################################
#csv = np.genfromtxt ('power.csv', delimiter=",",skip_header=1)
#X = csv[ 1:100, range(csv.shape[ 1 ] - 1) ]
#y = csv[ 1:100, csv.shape[ 1 ] - 1 ]

permutation = np.random.choice(range(X.shape[ 0 ]),
    X.shape[ 0 ], replace = False)
size_train = num_train
index_train = permutation[ 0 : size_train ]
index_test = permutation[ size_train : ]

X_train = X[ index_train, : ]
y_train = y[ index_train ]
X_test,y_test = generate_xy(rng,num_test)
plt.plot(X_test,y_test)
#plt.show()

#MC_net.MC_net(X_train,y_train,6,lam=10)
# MC_net.MC_net(X_train,y_train,1,lam=10)

net = EP_net.EP_net(X_train, y_train,[1],10,1)
net.train(X_train,y_train,40)
# m, v, v_noise = net.predict(X_test)
# plt.plot(X_test,m-7*np.sqrt(v),X_test,m+7*np.sqrt(v))
# plt.show()
#for (mean,var) in zip(m,v): 
    #print str(mean+5*np.sqrt(var)) +','+str(mean)+','+str(mean-5*np.sqrt(var))

