import lasagne
import numpy as np
import theano
import theano.tensor as T

from sklearn import metrics

import layers
import params
import util


class NodeClassificationDCNN(object):
    """A DCNN model for node classification.

    This is a shallow model.

    (K, X) -> DCNN -> Dense -> Out
    """
    def __init__(self, parameters, A):
        self.params = parameters

        self.var_K = T.tensor3('Apow')
        self.var_X = T.matrix('X')
        self.var_Y = T.imatrix('Y')

        self.l_in_k = lasagne.layers.InputLayer((None, self.params.num_hops + 1, self.params.num_nodes), input_var=self.var_K)
        self.l_in_x = lasagne.layers.InputLayer((self.params.num_nodes, self.params.num_features), input_var=self.var_X)

        self._compute_diffusion_kernel(A)

        # Overridable to customize init behavior.
        self._register_model_layers()

        loss_fn = params.loss_map[self.params.loss_fn]
        update_fn = params.update_map[self.params.update_fn]

        prediction = lasagne.layers.get_output(self.l_out)
        self._loss = lasagne.objectives.aggregate(loss_fn(prediction, self.var_Y), mode='mean')
        model_parameters = lasagne.layers.get_all_params(self.l_out)
        self._updates = update_fn(self._loss, model_parameters, learning_rate=self.params.learning_rate)
        if self.params.momentum:
            self._updates = lasagne.updates.apply_momentum(self._updates, model_parameters)

        self.apply_loss_and_update = theano.function([self.var_K, self.var_X, self.var_Y], self._loss, updates=self._updates)
        self.apply_loss = theano.function([self.var_K, self.var_X, self.var_Y], self._loss)

    def _compute_diffusion_kernel(self, A):
        self.K = util.A_to_diffusion_kernel(A, self.params.num_hops)

    def _register_model_layers(self):
        self.l_dcnn = layers.DCNNLayer(
            [self.l_in_k, self.l_in_x],
            self.params,
            1,
        )

        self.l_out = lasagne.layers.DenseLayer(
            self.l_dcnn,
            num_units=self.params.num_classes,
            nonlinearity=params.nonlinearity_map[self.params.out_nonlinearity],
        )

    def train_step(self, X, Y, batch_indices):
        return self.apply_loss_and_update(
            self.K[batch_indices, :, :], X, Y[batch_indices, :]
        )

    def validation_step(self, X, Y, valid_indices):
        return self.apply_loss(
            self.K[valid_indices, :, :], X, Y[valid_indices, :]
        )

    def fit(self, X, Y, train_indices, valid_indices):

        num_nodes = X.shape[0]

        print 'Training model...'
        validation_losses = []
        validation_loss_window = np.zeros(self.params.stop_window_size)
        validation_loss_window[:] = float('+inf')

        for epoch in range(self.params.num_epochs):
            train_loss = 0.0

            np.random.shuffle(train_indices)

            num_batch = num_nodes // self.params.batch_size

            for batch in range(num_batch):
                start = batch * self.params.batch_size
                end = min((batch + 1) * self.params.batch_size, train_indices.shape[0])

                if start < end:
                    train_loss += self.train_step(X, Y, train_indices[start:end])

            train_loss /= num_batch

            valid_loss = self.validation_step(X, Y, valid_indices)

            print "Epoch %d mean training error: %.6f" % (epoch, train_loss)
            print "Epoch %d validation error: %.6f" % (epoch, valid_loss)

            if self.params.print_train_accuracy:
                predictions = self.predict(X, train_indices)
                actuals = Y[train_indices, :].argmax(1)

                print "Epoch %d training accuracy: %.4f" % (epoch, metrics.accuracy_score(predictions, actuals))

            if self.params.print_valid_accuracy:
                predictions = self.predict(X, valid_indices)
                actuals = Y[valid_indices, :].argmax(1)

                print "Epoch %d validation accuracy: %.4f" % (epoch, metrics.accuracy_score(predictions, actuals), )

            validation_losses.append(valid_loss)

            if self.params.stop_early:
                if valid_loss >= validation_loss_window.mean():
                    print 'Validation loss did not decrease. Stopping early.'
                    break

            validation_loss_window[epoch % self.params.stop_window_size] = valid_loss

    def predict(self, X, prediction_indices):
        pred = lasagne.layers.get_output(self.l_out)

        # Create a function that applies the model to data to predict a class
        pred_fn = theano.function([self.var_K, self.var_X], T.argmax(pred, axis=1))

        # Return the predictions
        predictions = pred_fn(self.K[prediction_indices, :, :], X)

        return predictions


class TrueSparseNodeClassificationDCNN(NodeClassificationDCNN):
    """A DCNN model for node classification with truly sparse pre-thresholding.

    This is a shallow model.

    (K, X) -> DCNN -> Dense -> Out
    """

    def _compute_diffusion_kernel(self, A):
        self.K = util.sparse_A_to_diffusion_kernel(
            A,
            self.params.num_hops
        )

    def _register_model_layers(self):
        input = self.l_in_k + [self.l_in_x]
        self.l_dcnn = layers.SparseDCNNLayer(
            input,
            self.params,
            1,
        )

        self.l_out = lasagne.layers.DenseLayer(
            self.l_dcnn,
            num_units=self.params.num_classes,
            nonlinearity=params.nonlinearity_map[self.params.out_nonlinearity],
        )

    def __init__(self, parameters, A):
        self.params = parameters

        self.var_K = []
        for i in range(self.params.num_hops + 1):
            self.var_K.append(T.matrix('K_%d' % i))

        self.var_X = T.matrix('X')
        self.var_Y = T.imatrix('Y')

        self.l_in_k = [lasagne.layers.InputLayer((None, self.params.num_nodes), input_var=vK) for vK in self.var_K]
        self.l_in_x = lasagne.layers.InputLayer((self.params.num_nodes, self.params.num_features), input_var=self.var_X)

        self._compute_diffusion_kernel(A)

        # Overridable to customize init behavior.
        self._register_model_layers()

        loss_fn = params.loss_map[self.params.loss_fn]
        update_fn = params.update_map[self.params.update_fn]

        prediction = lasagne.layers.get_output(self.l_out)
        self._loss = lasagne.objectives.aggregate(loss_fn(prediction, self.var_Y), mode='mean')
        model_parameters = lasagne.layers.get_all_params(self.l_out)
        self._updates = update_fn(self._loss, model_parameters, learning_rate=self.params.learning_rate)
        if self.params.momentum:
            self._updates = lasagne.updates.apply_momentum(self._updates, model_parameters)

        self.apply_loss_and_update = theano.function(self.var_K + [self.var_X, self.var_Y], self._loss, updates=self._updates)
        self.apply_loss = theano.function(self.var_K + [self.var_X, self.var_Y], self._loss)

    def train_step(self, X, Y, batch_indices):
        #inputs = [k[batch_indices, :] for k in self.K] + [X, Y[batch_indices, :]]
        inputs = self.K + [X, Y[batch_indices, :]]
        return self.apply_loss_and_update(
            *inputs
        )

    def validation_step(self, X, Y, valid_indices):
        return self.apply_loss(
            self.K[valid_indices, :, :], X, Y[valid_indices, :]
        )


class PostSparseNodeClassificationDCNN(NodeClassificationDCNN):
    def _compute_diffusion_kernel(self, A):
        self.K = util.A_to_post_sparse_diffusion_kernel(
            A,
            self.params.num_hops,
            self.params.diffusion_threshold
        )


class PreSparseNodeClassificationDCNN(NodeClassificationDCNN):
    def _compute_diffusion_kernel(self, A):
        self.K = util.A_to_pre_sparse_diffusion_kernel(
            A,
            self.params.num_hops,
            self.params.diffusion_threshold
        )


class DeepNodeClassificationDCNN(NodeClassificationDCNN):
    """A Deep DCNN model for node classification.

    This model allows for several DCNN layers.

    (K, X) -> DCNN -> DCNN -> ... -> DCNN -> Dense -> Out
    """
    def __init__(self, parameters, A):
        self.params = parameters

        # Prepare indices input.
        self.var_K = T.tensor3('Apow')
        self.var_X = T.matrix('X')
        self.var_I = T.ivector('I')
        self.var_Y = T.imatrix('Y')

        self.l_in_k = lasagne.layers.InputLayer((None, self.params.num_hops + 1, self.params.num_nodes),
                                                input_var=self.var_K)
        self.l_in_x = lasagne.layers.InputLayer((self.params.num_nodes, self.params.num_features), input_var=self.var_X)
        self.l_indices = lasagne.layers.InputLayer(
            (None,),
            input_var=self.var_I
        )

        self.K = util.A_to_diffusion_kernel(A, self.params.num_hops)

        # Overridable to customize init behavior.
        self._register_model_layers()

        loss_fn = params.loss_map[self.params.loss_fn]
        update_fn = params.update_map[self.params.update_fn]

        prediction = lasagne.layers.get_output(self.l_out)
        self._loss = lasagne.objectives.aggregate(loss_fn(prediction, self.var_Y), mode='mean')
        model_parameters = lasagne.layers.get_all_params(self.l_out)
        self._updates = update_fn(self._loss, model_parameters, learning_rate=self.params.learning_rate)
        if self.params.momentum:
            self._updates = lasagne.updates.apply_momentum(self._updates, model_parameters)

        self.apply_loss_and_update = theano.function([self.var_K, self.var_X, self.var_I, self.var_Y], self._loss,
                                                     updates=self._updates)
        self.apply_loss = theano.function([self.var_K, self.var_X, self.var_I, self.var_Y], self._loss)

    def _register_model_layers(self):
        features_layer = self.l_in_x
        num_features = self.params.num_features

        for i in range(self.params.num_dcnn_layers):
            l_dcnn = layers.DCNNLayer(
                [self.l_in_k, features_layer],
                self.params,
                i + 1,
                num_features=num_features,
            )

            num_features *= (self.params.num_hops + 1)
            features_layer = lasagne.layers.ReshapeLayer(
                l_dcnn,
                (-1, num_features)
            )


        self.l_slice = layers.ArrayIndexLayer(
            [features_layer, self.l_indices]
        )

        self.l_out = lasagne.layers.DenseLayer(
            self.l_slice,
            num_units=self.params.num_classes,
            nonlinearity=params.nonlinearity_map[self.params.out_nonlinearity],
        )

    def train_step(self, X, Y, batch_indices):
        return self.apply_loss_and_update(
            self.K, X, batch_indices, Y[batch_indices, :]
        )

    def validation_step(self, X, Y, valid_indices):
        return self.apply_loss(
            self.K, X, valid_indices, Y[valid_indices, :]
        )

    def predict(self, X, prediction_indices):
        pred = lasagne.layers.get_output(self.l_out)

        # Create a function that applies the model to data to predict a class
        pred_fn = theano.function([self.var_K, self.var_X, self.var_I], T.argmax(pred, axis=1))

        # Return the predictions
        predictions = pred_fn(self.K, X, prediction_indices)

        return predictions


class DeepDenseNodeClassificationDCNN(NodeClassificationDCNN):
    """A Deep DCNN model for node classification.

    Composed of one DCNN layer for the input followed by several dense layers.

    (K, X) -> DCNN -> Dense -> Dense -> ... -> Dense -> Out
    """
    def _register_model_layers(self):
        self.l_dcnn = layers.DCNNLayer(
            [self.l_in_k, self.l_in_x],
            self.params,
            1,
        )

        input = self.l_dcnn

        for i in range(self.params.num_dense_layers):
            l_dense = lasagne.layers.DenseLayer(
                input,
                num_units=self.params.dense_layer_size,
                nonlinearity=params.nonlinearity_map[self.params.dense_nonlinearity],
            )
            input = l_dense


        self.l_out = lasagne.layers.DenseLayer(
            input,
            num_units=self.params.num_classes,
            nonlinearity=params.nonlinearity_map[self.params.out_nonlinearity],
        )


class GraphClassificationDCNN(object):
    """A DCNN for graph classification.

    DCNN Activations are mean-reduced across nodes.

    (P, X) -> DCNN -> Dense -> Out
    """
    def __init__(self, parameters):
        self.params = parameters

        self.var_A = T.matrix('A')
        self.var_X = T.matrix('X')
        self.var_Y = T.imatrix('Y')

        self.l_in_a = lasagne.layers.InputLayer((None, None), input_var=self.var_A)
        self.l_in_x = lasagne.layers.InputLayer((None, self.params.num_features), input_var=self.var_X)

        # Overridable to customize init behavior.
        self._register_model_layers()

        loss_fn = params.loss_map[self.params.loss_fn]
        update_fn = params.update_map[self.params.update_fn]

        prediction = lasagne.layers.get_output(self.l_out)
        loss = lasagne.objectives.aggregate(loss_fn(prediction, self.var_Y), mode='mean')
        model_parameters = lasagne.layers.get_all_params(self.l_out)
        self._updates = update_fn(loss, model_parameters, learning_rate=self.params.learning_rate)
        if self.params.momentum:
            self._updates = lasagne.updates.apply_momentum(self._updates, model_parameters)

        self.apply_loss_and_update = theano.function([self.var_A, self.var_X, self.var_Y], loss, updates=self._updates)
        self.apply_loss = theano.function([self.var_A, self.var_X, self.var_Y], loss)

        pred = lasagne.layers.get_output(self.l_out)
        self.pred_fn = theano.function([self.var_A, self.var_X], T.argmax(pred, axis=1))

    def _register_model_layers(self):
        self.l_dcnn = layers.AggregatedDCNNLayer(
            [self.l_in_a, self.l_in_x],
            self.params,
            1,
        )

        self.l_out = lasagne.layers.DenseLayer(
            self.l_dcnn,
            num_units=self.params.num_classes,
            nonlinearity=params.nonlinearity_map[self.params.out_nonlinearity],
        )

    def train_step(self, a, x, y):
        return self.apply_loss_and_update(
            a, x, y
        )

    def validation_step(self, a, x, y):
        return self.apply_loss(
            a, x, y
        )

    def fit(self, A, X, Y, train_indices, valid_indices):
        print 'Training model...'
        validation_losses = []
        validation_loss_window = np.zeros(self.params.stop_window_size)
        validation_loss_window[:] = float('+inf')

        for epoch in range(self.params.num_epochs):
            np.random.shuffle(train_indices)

            train_loss = 0.0

            for index in train_indices:
                train_loss += self.train_step(A[index], X[index], Y[index])
            train_loss /= len(train_indices)

            valid_loss = 0.0
            for index in valid_indices:
                valid_loss = self.validation_step(A[index], X[index], Y[index])
            valid_loss /= len(valid_indices)

            print "Epoch %d mean training error: %.6f" % (epoch, train_loss)
            print "Epoch %d mean validation error: %.6f" % (epoch, valid_loss)

            if np.isnan(train_loss) or np.isnan(valid_loss):
                raise ValueError

            train_acc = 0.0
            if self.params.print_train_accuracy:
                for index in train_indices:
                    pred = self.predict(A[index], X[index])
                    actual = Y[index].argmax()
                    if pred == actual:
                        train_acc += 1.0
                train_acc /= len(train_indices)
                print "Epoch %d training accuracy: %.4f" % (epoch, train_acc)

            valid_acc = 0.0
            if self.params.print_valid_accuracy:
                for index in valid_indices:
                    pred = self.predict(A[index], X[index])
                    actual = Y[index].argmax()
                    if pred == actual:
                        valid_acc += 1.0
                valid_acc /= len(train_indices)
                print "Epoch %d validation accuracy: %.4f" % (epoch, valid_acc)

            validation_losses.append(valid_loss)

            if self.params.stop_early:
                if valid_loss >= validation_loss_window.mean():
                    print 'Validation loss did not decrease. Stopping early.'
                    break

            validation_loss_window[epoch % self.params.stop_window_size] = valid_loss

    def predict(self, a, x):
        # Return the predictions
        predictions = self.pred_fn(a, x)

        return predictions


class GraphClassificationFeatureAggregatedDCNN(GraphClassificationDCNN):
    """A DCNN for graph classification.

    DCNN Activations are mean-reduced across both nodes and features.

    (P, X) -> DCNN -> Dense -> Out
    """
    def _register_model_layers(self):
        self.l_dcnn = layers.AggregatedFeaturesDCNNLayer(
            [self.l_in_a, self.l_in_x],
            self.params,
            1,
        )

        self.l_out = lasagne.layers.DenseLayer(
            self.l_dcnn,
            num_units=self.params.num_classes,
            nonlinearity=params.nonlinearity_map[self.params.out_nonlinearity],
        )


class DeepGraphClassificationDCNN(GraphClassificationDCNN):
    """A Deep DCNN for graph classification.

    DCNN Activations are mean-reduced across nodes.  Several DCNN layers.

    (P, X) -> DCNN -> DCNN -> ... -> DCNN -> Dense -> Out
    """
    def _register_model_layers(self):
        features_layer = self.l_in_x
        num_features = self.params.num_features

        for i in range(self.params.num_dcnn_layers - 1):
            l_dcnn = layers.UnaggregatedDCNNLayer(
                [self.l_in_a, features_layer],
                self.params,
                i + 1,
                num_features=num_features
            )
            features_layer = l_dcnn
            num_features *= (self.params.num_hops + 1)

        l_dcnn = layers.AggregatedDCNNLayer(
            [self.l_in_a, features_layer],
            self.params,
            i + 1,
            num_features=num_features,
        )

        self.l_out = lasagne.layers.DenseLayer(
            l_dcnn,
            num_units=self.params.num_classes,
            nonlinearity=params.nonlinearity_map[self.params.out_nonlinearity],
        )


class DeepGraphClassificationDCNNWithReduction(DeepGraphClassificationDCNN):
    """A Deep DCNN for graph classification with a trivial reduction layer.

        DCNN Activations are mean-reduced across nodes.  Several DCNN layers.

        (P, X) -> DCNN -> Reduction -> DCNN -> ... -> DCNN -> Dense -> Out
        """

    def _register_model_layers(self):
        graph_layer = self.l_in_a
        features_layer = self.l_in_x
        num_features = self.params.num_features

        for i in range(self.params.num_dcnn_layers - 1):
            l_dcnn = layers.UnaggregatedDCNNLayer(
                [graph_layer, features_layer],
                self.params,
                i,
                num_features=num_features
            )
            features_layer = l_dcnn
            num_features *= (self.params.num_hops + 1)

            graph_layer = layers.GraphReductionLayer(
                [graph_layer, features_layer],
                self.params,
            )

        l_dcnn = layers.AggregatedDCNNLayer(
            [graph_layer, features_layer],
            self.params,
            i,
            num_features=num_features,
        )

        self.l_out = lasagne.layers.DenseLayer(
            l_dcnn,
            num_units=self.params.num_classes,
            nonlinearity=params.nonlinearity_map[self.params.out_nonlinearity]
        )


class DeepGraphClassificationDCNNWithKronReduction(DeepGraphClassificationDCNN):
    """A Deep DCNN for graph classification with a learnable reduction layer.

        DCNN Activations are mean-reduced across nodes.  Several DCNN layers.

        (P, X) -> DCNN -> Reduction -> DCNN -> ... -> DCNN -> Dense -> Out
        """

    def _register_model_layers(self):
        graph_layer = self.l_in_a
        features_layer = self.l_in_x
        num_features = self.params.num_features

        for i in range(self.params.num_dcnn_layers - 1):
            l_dcnn = layers.UnaggregatedDCNNLayer(
                [graph_layer, features_layer],
                self.params,
                i,
                num_features=num_features
            )
            features_layer = l_dcnn
            num_features *= (self.params.num_hops + 1)

            eigenvec_layer = layers.SmallestEigenvecLayer(
                [graph_layer],
                self.params
            )

            graph_layer = layers.KronReductionLayerA(
                [graph_layer, eigenvec_layer],
                self.params,
            )

            features_layer = layers.KronReductionLayerX(
                [graph_layer, features_layer, eigenvec_layer],
                self.params,
            )

        l_dcnn = layers.AggregatedDCNNLayer(
            [graph_layer, features_layer],
            self.params,
            i,
            num_features=num_features,
        )

        self.l_out = lasagne.layers.DenseLayer(
            l_dcnn,
            num_units=self.params.num_classes,
            nonlinearity=params.nonlinearity_map[self.params.out_nonlinearity]
        )

