
import numpy as np

from sklearn import cluster

class EM_net:
    def __init__(self, X_train, y_train, n_hidden,lam=1,eta=0.01,a=0.0001):
        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)
        self.mean_X_train = np.mean(X_train, 0)
        self.n_units = n_hidden
        self.c = cluster.KMeans(n_clusters=self.n_units).fit(X_train).cluster_centers_ 
        self.lam = lam
        #learning rate
        self.eta = eta
        #regularizer
        self.a = a
        #add bias in weights
        self.w_sgd = np.random.randn(self.n_units+1)
    def pseudo_inverse(self, X_train, y_train):
        G = []
        for x in X_train:
            basis_out = np.exp(-self.lam*np.sum((self.c-x)**2,axis=1))
            basis_out = np.append(basis_out,1)
            G.append(basis_out)
        G_inv = np.linalg.pinv(G)
        self.w_inv = np.dot(G_inv,y_train)
    def sgd(self, X_train, y_train,n_epochs=1):
        for i in range(n_epochs): 
            for x,y in zip(X_train,y_train):
                rbf_val = np.exp(-self.lam*np.sum((self.c-x)**2,axis=1))
                #add bias
                rbf_val = np.append(rbf_val,1)
                rbf_out = np.sum(self.w_sgd*rbf_val)
                self.w_sgd = (1-self.a)*self.w_sgd + self.eta*(y-rbf_out)*rbf_val
    def pseudo_predict(self, X_test):    
        rbf_val = []
        for x in X_test:
            basis_out = np.exp(-self.lam*np.sum((self.c-x)**2,axis=1))
            basis_out = np.append(basis_out,1)
            rbf_val.append(basis_out)
        
        rbf_out = np.sum(self.w_inv*rbf_val,axis=1)
        return rbf_out 
    def sgd_predict(self, X_test):    
        rbf_val = []
        for x in X_test:
            basis_out = np.exp(-self.lam*np.sum((self.c-x)**2,axis=1))
            basis_out = np.append(basis_out,1)
            rbf_val.append(basis_out)
        
        rbf_out = np.sum(self.w_sgd*rbf_val,axis=1)
        return rbf_out 
    def get_sgd_weights(self):
        return self.w_sgd

