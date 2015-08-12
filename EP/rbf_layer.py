import math

import theano

import theano.tensor as T
import theano.tensor.nlinalg as alg

class RBF_layer(Network_Layer):

    def __init__(self, m_c_init, v_c_init, non_linear = True):

        # We create the theano variables for the means and variances

        self.m_c = theano.shared(value = m_c_init.astype(theano.config.floatX),
            name='m_c', borrow = True)

        self.v_c = theano.shared(value = v_c_init.astype(theano.config.floatX),
            name='v_c', borrow = True)
        self.w = theano.shared(value = m_c_init.astype(theano.config.floatX),
            name='c', borrow = True)
        # We store the type of activation function

        self.non_linear = non_linear

        # We store the number of inputs

        self.n_inputs = theano.shared(int(m_c_init.shape[ 1 ]))

    @staticmethod
    def n_pdf(x):

        return 1.0 / T.sqrt(2 * math.pi) * T.exp(-0.5 * x**2)

    @staticmethod
    def n_cdf(x):

        return 0.5 * (1.0 + T.erf(x / T.sqrt(2.0)))

    @staticmethod
    def gamma(x):

        return Network_layer.n_pdf(x) / Network_layer.n_cdf(-x)

    @staticmethod
    def beta(x):

        return Network_layer.gamma(x) * (Network_layer.gamma(x) - x)

    def output_probabilistic(self, m_c_previous, v_c_previous):

        # We add an additional deterministic input with mean 1 and variance 0

        #m_c_previous_with_bias = \
            #T.concatenate([ m_c_previous, T.alloc(1, 1) ], 0)
        #v_c_previous_with_bias = \
            #T.concatenate([ v_c_previous, T.alloc(0, 1) ], 0)

        # We compute the mean and variance after the linear operation


        m_linear = self.m_c - m_c_previous

        #m_linear = T.dot(self.m_c, m_c_previous_with_bias) / T.sqrt(self.n_inputs)
        v_linear = self.v_c

        if (self.non_linear):

            # We compute the mean and variance after the RBF activation

            lam = 0.1
            v_1 = 1 + 2*lam*v_linear


            v_1_inv = v_1**-1
            s_1 = T.prod(v_1,axis=1)**-0.5

            v_2 = 1 + 4*lam*v_linear
            v_2_inv = v_2**-1


            s_2 = T.prod(v_2,axis=1)**-0.5



            v_inv = v_linear**-1





            exponent1 = m_linear*(1 - v_1_inv)*v_inv

            exponent1 = T.sum(exponent1,axis=1)

            exponent2 = m_linear**2*(1 - v_2_inv)*v_inv

            exponent2 = T.sum(exponent2,axis=1)


            m_a = s_1*T.exp(-0.5*exponent1)


            v_a = s_2*T.exp(-0.5*exponent2) - m_a**2


    #        lam = 1
    #        I = T.eye(self.n_inputs)
#
    #        v_1 = I + 2*lam*v_linear
    #        print_op= theano.printing.Print('v_1')
    #        v_1 = print_op(v_1)
#
    #        def compute_inverse(v):
    #            return alg.MatrixInverse()(v)
#
#
    #        v_1_inv,update = theano.map(compute_inverse, sequences=v_1)
    #        s_1,update = theano.map(lambda v: alg.Det()(v), sequences=v_1)
    #        s_1 = s_1**-0.5
#
#
    #        v_2 = I + 4*lam*v_linear
    #        v_2_inv,update = theano.map(lambda v: alg.MatrixInverse()(v), sequences=v_2)
    #        s_2,update = theano.map(lambda v: alg.Det()(v), sequences=v_2)
#
    #        s_2 = s_2**-0.5
    #        v_inv,update = theano.map(lambda v: alg.MatrixInverse()(v), sequences=v_linear)
#
    #        m_a = s_1*T.exp(-0.5*m_linear.T*(I - v_1_inv)*v_inv*m_linear)
#
    #        v_a = s_2*T.exp(-0.5*m_linear.T*(I - v_2_inv)*v_inv*m_linear - m_a**2)
            return (m_a, v_a)

        else:

            return (m_linear, v_linear)

    def output_deterministic(self, output_previous):

        # We add an additional input with value 1

        output_previous_with_bias = \
            T.concatenate([ output_previous, T.alloc(1, 1) ], 0) / \
            T.sqrt(self.n_inputs)

        # We compute the mean and variance after the linear operation

        a = T.dot(self.w, output_previous_with_bias)

        if (self.non_linear):

            # We compute the ReLU activation

            a = T.switch(T.lt(a, T.fill(a, 0)), T.fill(a, 0), a)

        return a
