import numpy as np

import theano
import theano.tensor as T



import math

import numpy as np

import sys
sys.path.append('PBP_net/')
import PBP_net
import matplotlib.pyplot as plt

np.random.seed(1)

# We load the boston housing dataset

#data = np.loadtxt('boston_housing.txt')

# We obtain the features and the targets

#X = data[ :, range(data.shape[ 1 ] - 1) ]
#y = data[ :, data.shape[ 1 ] - 1 ]


# We create the train and test sets with 90% and 10% of the data

#Generate artificial data
num_pts = 200
x_pts =  np.linspace(-20,20,num=num_pts)
x2_pts =  np.linspace(-5,5,num=num_pts)
X =  np.array([x_pts,x2_pts]).T
y = 0.5*x_pts**2 + x_pts + 0.3*+x_pts**3 + 0.1*x_pts**4 + x2_pts**5 + 5*np.random.randn(num_pts)



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
X_train = np.array([[1,1],[2,2],[3,3],[4,4]])
y_train = np.array([1,2,3,4])
print X.shape
print y.shape


n_hidden_units = 1
net = PBP_net.PBP_net(X_train, y_train,
    [n_hidden_units ], normalize = True, n_epochs = 100)

# We make predictions for the test set

#m, v, v_noise = net.predict(X_test)#

##plt.plot(X_test,y_test,'ro',X_test,m,'bx')
##plt.show()
## We compute the test RMSE#

#rmse = np.sqrt(np.mean((y_test - m)**2))#

#print 'rmse'
#print rmse#

## We compute the test log-likelihood#

#test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
#    0.5 * (y_test - m)**2 / (v + v_noise))
#print "test_log likelihood"
#print test_ll
