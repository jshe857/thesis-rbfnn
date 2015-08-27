#!/usr/bin/python2
import math
import numpy as np
import sys
sys.path.append('EM/')
import EM_net
sys.path.append('EP/')
import EP_net

# import matplotlib
# matplotlib.use('pgf')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
np.random.seed(1)


# We create the train and test sets with 90% and 10% of the data
#Generate artificial data
num_train = 1000
num_test = 25

def generate_xy(rng,num,noise=True):
    x_pts =  np.linspace(-rng,rng,num=num)
    X = np.array([x_pts]).T
    if (noise):
        y = 3*np.cos(x_pts/9) +  2*np.sin(x_pts/15) + 0.1*np.random.randn(num)
    else:
        y = 3*np.cos(x_pts/9) +  2*np.sin(x_pts/15)
    return(X,y)

rng = 80
X,y = generate_xy(rng,num_train)


x_true,y_true = generate_xy(rng,200,noise=False)



print X.shape
print y.shape
permutation = np.random.choice(range(X.shape[ 0 ]),
    X.shape[ 0 ], replace = False)
size_train = num_train
index_train = permutation[ 0 : size_train ]
index_test = permutation[ size_train : ]

X_train = X[ index_train, : ]
y_train = y[ index_train ]
# X_test = X[ index_test, : ]
# y_test = y[ index_test ]

X_test,y_test = generate_xy(rng,num_test)



# We construct the network with one hidden layer with two-hidden layers
# with 50 neurons in each one and normalizing the training features to have
# zero mean and unit standard deviation in the trainig set.
lam = 12
var_prior = 2
n_hidden_units = 200
net = EP_net.EP_net(X_train, y_train,
    [n_hidden_units],lam,var_prior)

#m, v, v_noise = net.predict(X_test)
#plt.plot(X_test,y_test,'ro',X_test,m,'bx')
#plt.show()

net.train(X_train,y_train,40)
m, v, v_noise = net.predict(X_test)
plt.plot(x_true,y_true,'r')
plt.errorbar(X_test,m,fmt='bx',yerr= 2*np.sqrt(v))
#plt.plot(X_train,y_train,'go')
red_patch = mpatches.Patch(color='red', label='True function')
blue_patch = mpatches.Patch(color='blue', label='Prediction')
#green_patch = mpatches.Patch(color='green', label='Training Data')
plt.ylabel('y')
plt.xlabel('x')
plt.title('n='+str(size_train))
plt.legend(handles=[red_patch,blue_patch],loc=2)
fig = plt.gcf()
plt.show()
fig.savefig('test_1d_train'+str(size_train)+'lam'+ str(lam)+'prior'+ str(var_prior)+'.eps', format='eps', dpi=500)
# fig.savefig('test_1d_train'+str(size_train)+'lam'+ str(lam)+'prior'+ str(var_prior)+'.eps')
# We make predictions for the test set
#skip_len = 10
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
print '====================EP========================'
print 'rmse'
print rmse

# We compute the test log-likelihood

test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
    0.5 * (y_test - m)**2 / (v + v_noise))
print "test_log likelihood"
print test_ll

################# EM for RBFNN approach #############################################
if (n_hidden_units > num_train): sys.exit() 
em = EM_net.EM_net(X_train,y_train, n_hidden_units,lam,eta=1.1,a=0.1)

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
plt.plot(X_test,y_test,'ro',X_test,rbf_sgd,'bx')
plt.show()
print '====================EM========================'
print 'test error'
print rmse

rbf_sgd = em.sgd_predict(X_train)
rmse = np.sqrt(np.mean((y_train - rbf_sgd)**2))
print 'train error'
print rmse
