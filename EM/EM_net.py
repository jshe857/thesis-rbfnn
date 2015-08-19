
import numpy as np

from sklearn import cluster

class EM_net:
    
    def __init__(self, X_train, y_train, n_hidden,lam=1):
        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)
        self.mean_X_train = np.mean(X_train, 0)
        self.n_units = n_hidden
        self.c = cluster.KMeans(n_clusters=self.n_units).fit(X_train).cluster_centers_ 
        self.lam = lam 
        self.eta = 1.0
        self.w_sgd = np.random.randn(self.n_units)
    def pseudoInverse(self, X_train, y_train):
        G = []
        for x in X_train:
            G.append(np.exp(-self.lam*np.sum((self.c-x)**2,axis=1)))
        G = np.array(G)
        G_inv = np.linalg.pinv(G)
        self.w_inv = np.dot(G_inv,y_train)
    def sgd(self, X_train, y_train):
        for x,y in zip(X_train,y_train):
            rbf_val = np.exp(-self.lam*np.sum((self.c-x)**2,axis=1))
            rbf_out = np.sum(self.w_sgd*rbf_val)
            self.w_sgd = self.w_sgd + self.eta*(y-rbf_out)*rbf_val
    def pseudoPredict(self, X_test):    
        rbf_val = []
        for x in X_test:
            rbf_val.append(np.exp(-self.lam*np.sum((self.c-x)**2,axis=1)))
        
        rbf_out = np.sum(self.w_inv*rbf_val,axis=1)
        return rbf_out 
    def sgdPredict(self, X_test):    
        rbf_val = []
        for x in X_test:
            rbf_val.append(np.exp(-self.lam*np.sum((self.c-x)**2,axis=1)))
        
        rbf_out = np.sum(self.w_sgd*rbf_val,axis=1)
        return rbf_out 

