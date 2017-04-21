import unittest

import numpy as np
import theano.tensor as T
import lasagne.layers

import layers
import params

class TestDCNNInternals(unittest.TestCase):
    def setUp(self):
        self.params = params.Params(
            num_nodes = 10,
            num_features = 5,
            num_classes = 3,
            num_hops = 3,
            learning_rate = 0.1,
            dcnn_nonlinearity = 'sigmoid'
        )
        self.A = (np.ones((self.params.num_nodes, self.params.num_nodes)) - np.eye(self.params.num_nodes)).astype('float')

        self.X = np.random.randn(self.params.num_nodes, self.params.num_features).astype(dtype='float')

        self.Y = np.random.randint(0, 2, (self.params.num_nodes, self.params.num_classes), dtype='int32')

        self.var_A = T.matrix('testmodel_A')
        self.var_X = T.matrix('testmodel_X')
        self.var_Y = T.matrix('testmodel_Y')

        self.l_in_a = lasagne.layers.InputLayer((None, self.params.num_nodes), input_var=self.var_A)
        self.l_in_x = lasagne.layers.InputLayer((self.params.num_nodes, self.params.num_features), input_var=self.var_X)

        self.l_apow = layers.PowerSeriesLayer(self.l_in_a, self.params)

        self.dcnn = layers.DCNNLayer(
            [self.l_in_a, self.l_in_x],
            self.params,
            1
        )

    def test_output_shape(self):
        out_shape = self.dcnn.get_output_shape_for([
            self.A.shape,
            self.X.shape
        ])

        self.assertEqual(out_shape[0], None)
        self.assertEqual(out_shape[1], self.params.num_hops + 1)
        self.assertEqual(out_shape[2], self.params.num_features)

    def test_output(self):
        out = self.dcnn.get_output_for([
            self.var_A,
            self.var_X
        ])

        out_shape = out.shape.eval({
            self.var_A: self.A,
            self.var_X: self.X,
        })

        self.assertEqual(out_shape[0], self.params.num_nodes)
        self.assertEqual(out_shape[1], self.params.num_hops + 1)
        self.assertEqual(out_shape[2], self.params.num_features)
