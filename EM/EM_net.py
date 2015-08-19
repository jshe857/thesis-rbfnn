
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
    def train(self, X_train, y_train):
        G = []
        for x in X_train:
            G.append(np.exp(-self.lam*np.sum((self.c-x)**2,axis=1)))
        G = np.array(G)
        G_inv = np.linalg.pinv(G)
        self.w = np.dot(G_inv,y_train)
        print 'starting grad descent....'
        self.w2 = np.random.randn(self.n_units)
        for x,y in zip(X_train,y_train):
            rbf_val = np.exp(-self.lam*np.sum((self.c-x)**2,axis=1))
            rbf_out = np.sum(self.w*rbf_val)
            self.w2 = self.w2 + self.eta*(y-rbf_out)*rbf_val
    def predict(self, X_test):    
        rbf_val = []
        for x in X_test:
            rbf_val.append(np.exp(-self.lam*np.sum((self.c-x)**2,axis=1)))
        
        rbf_pseudo = np.sum(self.w*rbf_val,axis=1)
        rbf_sgd = np.sum(self.w2*rbf_val,axis=1)
        return (rbf_sgd,rbf_pseudo)
