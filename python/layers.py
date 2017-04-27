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

    def get_output_for(self, inputs):
        pass

    def get_output_shape_for(self, input_shapes):
        pass

