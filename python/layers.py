import lasagne
import lasagne.layers
import theano
import theano.tensor as T
import numpy as np

import params


class DCNNLayer(lasagne.layers.MergeLayer):
    """A node-level DCNN layer.

    This class contains the (symbolic) Lasagne internals for a node-level DCNN layer.  This class should
    be used in conjunction with a user-facing model class.
    """
    def __init__(self, incomings, parameters, layer_num,
                 W=lasagne.init.Normal(0.01),
                 num_features=None,
                 **kwargs):

        super(DCNNLayer, self).__init__(incomings, **kwargs)

        self.parameters = parameters

        if num_features is None:
            self.num_features = self.parameters.num_features
        else:
            self.num_features = num_features

        self.W = T.addbroadcast(
            self.add_param(W,
                           (1, parameters.num_hops + 1, self.num_features), name='DCNN_W_%d' % layer_num), 0)

        self.nonlinearity = params.nonlinearity_map[self.parameters.dcnn_nonlinearity]

    def get_output_for(self, inputs, **kwargs):
        """Compute diffusion convolutional activation of inputs."""

        Apow = inputs[0]

        X = inputs[1]

        Apow_dot_X = T.dot(Apow, X)

        Apow_dot_X_times_W = Apow_dot_X * self.W

        out = self.nonlinearity(Apow_dot_X_times_W)

        return out

    def get_output_shape_for(self, input_shapes):
        """Return the layer output shape."""

        shape = (None, self.parameters.num_hops + 1, self.num_features)
        return shape

class SparseDCNNLayer(DCNNLayer):
    """A node-level DCNN layer.

    This class contains the (symbolic) Lasagne internals for a node-level DCNN layer.  This class should
    be used in conjunction with a user-facing model class.
    """
    def __init__(self, incomings, parameters, layer_num,
                 W=lasagne.init.Normal(0.01),
                 num_features=None,
                 **kwargs):

        super(DCNNLayer, self).__init__(incomings, **kwargs)

        self.parameters = parameters

        if num_features is None:
            self.num_features = self.parameters.num_features
        else:
            self.num_features = num_features

        self.W = T.addbroadcast(
            self.add_param(W,
                           (1, parameters.num_hops + 1, self.num_features), name='DCNN_W_%d' % layer_num), 0)

        self.nonlinearity = params.nonlinearity_map[self.parameters.dcnn_nonlinearity]


    def get_output_for(self, inputs, **kwargs):
        """Compute diffusion convolutional activation of inputs."""

        Apow = T.horizontal_stack(*inputs[:-1])

        X = inputs[-1]

        Apow_dot_X = T.dot(Apow, X)

        Apow_dot_X_times_W = Apow_dot_X * self.W

        out = self.nonlinearity(Apow_dot_X_times_W)

        return out


class AggregatedDCNNLayer(lasagne.layers.MergeLayer):
    """A graph-level DCNN layer.

    This class contains the (symbolic) Lasagne internals for a graph-level DCNN layer.  This class should
    be used in conjunction with a user-facing model class.
    """

    def __init__(self, incomings, parameters, layer_num,
                 W=lasagne.init.Normal(0.01),
                 num_features=None,
                 **kwargs):
        super(AggregatedDCNNLayer, self).__init__(incomings, **kwargs)

        self.parameters = parameters

        if num_features is None:
            self.num_features = self.parameters.num_features
        else:
            self.num_features = num_features

        self.W = T.addbroadcast(
            self.add_param(W, (self.parameters.num_hops + 1, 1, self.num_features), name='AGGREGATE_DCNN_W_%d' % layer_num), 1)

        self.nonlinearity = params.nonlinearity_map[self.parameters.dcnn_nonlinearity]

    def get_output_for(self, inputs, **kwargs):
        """
        Compute diffusion convolution of inputs.

        """

        A = inputs[0]
        X = inputs[1]

        # Normalize by degree.
        A = A / (T.sum(A, 0) + 1.0)

        Apow_list = [T.identity_like(A)]
        for i in range(1, self.parameters.num_hops + 1):
            Apow_list.append(A.dot(Apow_list[-1]))
        Apow = T.stack(Apow_list)

        Apow_dot_X = T.dot(Apow, X)

        Apow_dot_X_times_W = Apow_dot_X * self.W

        out = T.reshape(
            self.nonlinearity(T.mean(Apow_dot_X_times_W, 1)),
            (1, (self.parameters.num_hops + 1) * self.num_features)
        )

        return out

    def get_output_shape_for(self, input_shapes):
        shape = (1, (self.parameters.num_hops + 1) * self.num_features)
        return shape


class UnaggregatedDCNNLayer(AggregatedDCNNLayer):
    """A graph-level DCNN layer.

    This class contains the (symbolic) Lasagne internals for a graph-level DCNN layer.  This class should
    be used in conjunction with a user-facing model class.
    """
    def get_output_for(self, inputs, **kwargs):
        """
        Compute diffusion convolution of inputs.

        """

        A = inputs[0]
        X = inputs[1]

        # Normalize by degree.
        A = A / (T.sum(A, 0) + 1.0)

        Apow_list = [T.identity_like(A)]
        for i in range(1, self.parameters.num_hops + 1):
            Apow_list.append(A.dot(Apow_list[-1]))
        Apow = T.stack(Apow_list)

        Apow_dot_X = T.dot(Apow, X)

        Apow_dot_X_times_W = Apow_dot_X * self.W

        out = T.reshape(
            self.nonlinearity(Apow_dot_X_times_W).transpose((1, 0, 2)),
            (A.shape[0], (self.parameters.num_hops + 1) * self.num_features)
        )

        return out

    def get_output_shape_for(self, input_shapes):
        num_nodes = input_shapes[0][0]
        shape = (num_nodes, (self.parameters.num_hops + 1) * self.num_features)
        return shape


class AggregatedFeaturesDCNNLayer(AggregatedDCNNLayer):
    """A graph-level DCNN layer that aggregates across features.

    This class contains the (symbolic) Lasagne internals for a graph-level DCNN layer.  This class should
    be used in conjunction with a user-facing model class.
    """

    def get_output_for(self, inputs, **kwargs):
        """
        Compute diffusion convolution of inputs.

        """

        A = inputs[0]
        X = inputs[1]

        # Normalize by degree.
        A = A / (T.sum(A, 0) + 1.0)

        Apow_list = [T.identity_like(A)]
        for i in range(1, self.parameters.num_hops + 1):
            Apow_list.append(A.dot(Apow_list[-1]))
        Apow = T.stack(Apow_list)

        Apow_dot_X = T.dot(Apow, X)

        Apow_dot_X_times_W = Apow_dot_X * self.W

        out = self.nonlinearity(
            T.mean(
                T.reshape(
                    T.mean(Apow_dot_X_times_W, 1),
                    (1, (self.parameters.num_hops + 1), self.num_features)
                ),
                2
            )
        )

        return out

    def get_output_shape_for(self, input_shapes):
        shape = (1, self.parameters.num_hops + 1)
        return shape


class ArrayIndexLayer(lasagne.layers.MergeLayer):
    def get_output_for(self, inputs, **kwargs):
        X = inputs[0]
        indices = inputs[1]

        return X[indices]

    def get_output_shape_for(self, input_shapes):
        in_shape = input_shapes[0]
        index_shape = input_shapes[1]
        shape = tuple([index_shape[0]] + list(in_shape[1:]))
        return shape


class PowerSeriesLayer(lasagne.layers.Layer):
    def __init__(self, incoming, parameters):
        super(PowerSeriesLayer, self).__init__(incoming)

        self.parameters = parameters

    def get_output_for(self, incoming):
        A = incoming

        # Normalize by degree.
        # TODO: instead of adding 1.0, set 0.0 to 1.0
        A = A / (T.sum(A, 0) + 1.0)

        Apow_elements = [T.identity_like(A)]
        for i in range(1, self.parameters.num_hops + 1):
            Apow_elements.append(A.dot(Apow_elements[-1]))
        Apow = T.stack(Apow_elements)

        return Apow.dimshuffle([1, 0, 2])

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.paramaters.num_hops + 1, input_shape[1])


class GraphReductionLayer(lasagne.layers.MergeLayer):
    def __init__(self, incoming, parameters):
        super(GraphReductionLayer, self).__init__(incoming)
        self.parameters = parameters

    def reduce(self, A, indices):
        i = indices[0]
        j = indices[1]

        A = T.set_subtensor(A[i, :], A[i, :] + A[j, :])
        A = T.set_subtensor(A[:, i], A[:, i] + A[:, j])
        A = T.set_subtensor(A[j, :], T.zeros([A.shape[1]]))
        A = T.set_subtensor(A[:, j], T.zeros([A.shape[1]]))

        return A

    def get_output_for(self, inputs):
        A = inputs[0]
        X = inputs[1]

        max_degree_node = T.argmax(A.sum(0))
        min_degree_node = T.argmin(A.sum(0))

        return self.reduce(A, [max_degree_node, min_degree_node])

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]


class LearnableGraphReductionLayer(GraphReductionLayer):
    def __init__(self,
                 incoming,
                 parameters,
                 layer_num,
                 W=lasagne.init.Normal(0.01),
                 num_features=None):
        super(LearnableGraphReductionLayer, self).__init__(incoming, parameters)

        if num_features is None:
            self.num_features = self.parameters.num_features
        else:
            self.num_features = num_features

        self.W = self.add_param(
            W,
            (1, self.parameters.num_hops + 1, self.num_features), name='DCNN_REDUCTION_%d' % layer_num
        )

    def _outer_substract(self, x, y):
        z = x.dimshuffle(0, 1, 'x')
        z = T.addbroadcast(z, 2)
        return (z - y.T).dimshuffle(0, 2, 1)

    def _symbolic_triangles(self, A):
        """Computes the number of triangles involving any two nodes.

        A * A^2
        For more details, see
        http://stackoverflow.com/questions/40269150/number-of-triangles-involving-any-two-nodes

        """
        return A * T.dot(A, A)

    def _symbolic_arrows(self, A):
        """Computes the number of unclosed triangles involving any two nodes.

        (1 - A) A^2 + A (D + D^T - A^2 - 1)
        """
        # Compute and broadcast degree.
        num_nodes = A.shape[0]
        D = T.tile(T.sum(A, axis=1), (num_nodes, 1))

        return (
            (T.eye(num_nodes) - A) * T.dot(A, A) +
            A * (D + D.T - T.dot(A, A) - 2)
        )

    def get_output_for(self, inputs):
        A = inputs[0]
        X = inputs[1]

        num_nodes = A.shape[0]
        structural_symbolic_loss = T.addbroadcast(
            T.reshape(
                1 + A + self._symbolic_triangles(A) + self._symbolic_arrows(A),
                [num_nodes, num_nodes, 1]
            ),
            2
        )

        feature_symbolic_loss = (
            (self._outer_substract(X, X) ** 2) *
            T.addbroadcast(self.W, 0, 1)
        )

        unnormalized_logprobs = T.sum(
            structural_symbolic_loss + feature_symbolic_loss,
            2
        )

        flat_reduction_index = T.argmax(unnormalized_logprobs)

        return self.reduce(A, [
            flat_reduction_index // num_nodes,
            flat_reduction_index % num_nodes
        ])


def _2d_slice(M, ind1, ind2):
    temp = M[ind1, :]
    temp = temp[:, ind2]

    return temp


class SmallestEigenvecLayer(lasagne.layers.MergeLayer):
    def __init__(self, incoming, parameters):
        super(SmallestEigenvecLayer, self).__init__(incoming)
        self.parameters = parameters

    def get_output_for(self, inputs):
        A = inputs[0]

        eigenvals_eigenvecs = T.nlinalg.eig(A)

        smallest_eigenval_index = T.argmin(eigenvals_eigenvecs[0])
        smallest_eigenvec = eigenvals_eigenvecs[1][smallest_eigenval_index]

        return smallest_eigenvec

    def get_output_shape_for(self, input_shapes):
        A_shape = input_shapes[0]

        return A_shape[0]


class KronReductionLayerA(lasagne.layers.MergeLayer):
    def __init__(self, incoming, parameters):
        super(KronReductionLayerA, self).__init__(incoming)
        self.parameters = parameters

        self.shape = (None, None)

    def get_output_for(self, inputs):
        A = inputs[0]
        smallest_eigenvec = inputs[1]

        keep = smallest_eigenvec >= 0
        reduced = smallest_eigenvec < 0

        a = _2d_slice(A, keep, keep)
        b = _2d_slice(A, keep, reduced)
        c = _2d_slice(A, reduced, reduced)

        reduced_A = a - b.dot(T.nlinalg.pinv(c)).dot(b.T)

        self.shape = reduced_A.shape

        return reduced_A

    def get_output_shape_for(self, input_shapes):
        return self.shape


class KronReductionLayerX(lasagne.layers.MergeLayer):
    def __init__(self, incoming, parameters):
        super(KronReductionLayerX, self).__init__(incoming)
        self.parameters = parameters

        self.shape = (None, None)

    def get_output_for(self, inputs):
        A = inputs[0]
        X = inputs[1]
        smallest_eigenvec = inputs[2]

        keep = smallest_eigenvec >= 0
        reduced = smallest_eigenvec < 0

        b = _2d_slice(A, keep, reduced)
        c = _2d_slice(A, reduced, reduced)

        bx = X[keep, :]
        cx = X[reduced, :]

        reduced_X = bx - b.dot(T.nlinalg.pinv(c)).dot(cx)

        self.shape = reduced_X.shape

        return reduced_X

    def get_output_shape_for(self, input_shapes):
        return self.shape
