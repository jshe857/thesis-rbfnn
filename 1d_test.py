#!/usr/bin/python2
import math
import numpy as np
import sys
sys.path.append('MC')
sys.path.append('EP')
import MC_net
import EP_net
# import matplotlib
# matplotlib.use('pgf')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
np.random.seed(1)
sns.set_context('poster')
# We create the train and test sets with 90% and 10% of the data
#Generate artificial data
num_train = 100 
num_test = 500
def generate_xy(rng,num,noise=True):
    x_pts =  np.linspace(-rng,rng,num=num)
    X = np.array([x_pts]).T
    if (noise):
        y = 3*np.cos(x_pts/9) + 2*np.sin(x_pts/15) + 0.5*np.random.randn(num)
        # y = 3*np.exp(-10*(x_pts - 1)**2) + 1+ 0.01*np.random.randn(num) 
    else:
        y = 3*np.cos(x_pts/9) + 2*np.sin(x_pts/15)
        # y = 3*np.exp(-10*(x_pts - 1)**2) + 1
    return(X,y)
rng = 70
# rng = 5
X,y = generate_xy(rng,num_train)
print X.shape
print y.shape

permutation = np.random.choice(range(X.shape[ 0 ]),
    X.shape[ 0 ], replace = False)
size_train = num_train
index_train = permutation[ 0 : size_train ]
index_test = permutation[ size_train : ]

X_train = X[ index_train, : ]
y_train = y[ index_train ]
X_test,y_test = generate_xy(rng,num_test,noise=False)
plt.plot(X_test,y_test)
plt.show()

#MC_net.MC_net(X_train,y_train,6,lam=10)
# MC_net.MC_net(X_train,y_train,1,lam=10)

net = EP_net.EP_net(X_train, y_train,[50],4,0.008,debug=False)
net.train(X_train,y_train,20)


m, v, v_noise = net.predict(X_test)
X_test = np.squeeze(X_test)
upper = (m-2*np.sqrt(v))
lower = (m+2*np.sqrt(v))
plt.scatter(X_train,y_train,c='r',s=10,label="Training Data")
plt.plot(X_test,m, label="EP")
plt.fill_between(X_test,upper,lower,alpha=0.4)
plt.plot(X_test,y_test,label="True Function")
plt.legend()
plt.show()
# for (mean,var) in zip(m,v): 
    # print str(mean+5*np.sqrt(var)) +','+str(mean)+','+str(mean-5*np.sqrt(var))

