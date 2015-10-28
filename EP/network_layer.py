
import math

import theano

import theano.tensor as T

class Network_layer:

    def __init__(self, m_w_init, v_w_init, non_linear,lam):
        # print('layer')
        # print m_w_init
        # We create the theano variables for the means and variances
        self.m_w = theano.shared(value = m_w_init.astype(theano.config.floatX),
            name='m_w', borrow = True)

        self.v_w = theano.shared(value = v_w_init.astype(theano.config.floatX),
            name='v_w', borrow = True)
        self.w = theano.shared(value = m_w_init.astype(theano.config.floatX),
            name='w', borrow = True)

        # We store the type of activation function

        self.non_linear = non_linear

        # We store the number of inputs

        self.n_inputs = theano.shared(float(m_w_init.shape[ 1 ]))
        self.lam = lam

    def output_probabilistic(self, m_w_previous, v_w_previous):
        if (self.non_linear):
            m_in = self.m_w - m_w_previous
            v_in = self.v_w
            # We compute the mean and variance after the ReLU activation
            lam = self.lam
            v_1 = 1 + 2*lam*v_in
            v_1_inv = v_1**-1

            s_1 = T.prod(v_1,axis=1)**-0.5
            v_2 = 1 + 4*lam*v_in
            v_2_inv = v_2**-1
            s_2 = T.prod(v_2,axis=1)**-0.5
            v_inv = v_in**-1
            exponent1 = m_in**2*(1 - v_1_inv)*v_inv
            exponent1 = T.sum(exponent1,axis=1)
            exponent2 = m_in**2*(1 - v_2_inv)*v_inv
            exponent2 = T.sum(exponent2,axis=1)
            m_a = s_1*T.exp(-0.5*exponent1)
            v_a = s_2*T.exp(-0.5*exponent2) - m_a**2

            return (m_a, v_a)

        else:
            m_w_previous_with_bias = \
            T.concatenate([ m_w_previous, T.alloc(1, 1) ], 0)
            v_w_previous_with_bias = \
            T.concatenate([ v_w_previous, T.alloc(0, 1) ], 0)

            m_linear = T.dot(self.m_w, m_w_previous_with_bias) / T.sqrt(self.n_inputs)
            v_linear = (T.dot(self.v_w, v_w_previous_with_bias) + \
                T.dot(self.m_w**2, v_w_previous_with_bias) + \
                T.dot(self.v_w, m_w_previous_with_bias**2)) / self.n_inputs
            return (m_linear, v_linear)
