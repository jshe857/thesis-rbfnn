import numpy as np

import theano
import theano.tensor as T

x = T.dmatrix('x')

y = T.prod(x,axis=0)
y2 = T.sum(x,axis=1)
f = theano.function([x],y)
f2 = theano.function([x],y2)
inp = np.array([[1,1],[3,4]])
print inp
print 'prod x, axis=0: '
print f(inp)
print 'sum x, axis=1: '
print f2(inp)

inp = np.array([[1,1]])
print inp
print 'prod x, axis=0: '
print f(inp)
print 'sum x, axis=1: '
print f2(inp)
