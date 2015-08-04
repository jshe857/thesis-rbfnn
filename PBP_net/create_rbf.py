import theano
import theano.tensor as T
import rbf_layer
import numpy as np


rbf = rbf_layer.RBF_layer(np.array([[2,2]]),np.array([[2,1]]))


x = T.dvector('x')
y = T.scalar('y')

v_0 = T.zeros_like(x)

m,v= rbf.output_probabilistic(x,v_0)


f = theano.function([x],[m,v])
print f(np.array([0,0]))
