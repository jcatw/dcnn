import lasagne
import numpy as np
import theano
import theano.tensor as T

import layers
import params
import util


class NodeClassificationDCNN(object):
    def __init__(self, parameters, A):
        self.params = parameters

        self.var_K = T.tensor3('Apow')
        self.var_X = T.matrix('X')
        self.var_Y = T.imatrix('Y')

        self.l_in_k = lasagne.layers.InputLayer((None, self.params.num_hops + 1, self.params.num_nodes), input_var=self.var_K)
        self.l_in_x = lasagne.layers.InputLayer((self.params.num_nodes, self.params.num_features), input_var=self.var_X)

        self.K = util.A_to_diffusion_kernel(A, self.params.num_hops)

        # Overridable to customize init behavior.
        self._register_model_layers()

        loss_fn = params.loss_map[self.params.loss_fn]
        update_fn = params.update_map[self.params.update_fn]

        prediction = lasagne.layers.get_output(self.l_out)
        self._loss = lasagne.objectives.aggregate(loss_fn(prediction, self.var_Y), mode='mean')
        model_parameters = lasagne.layers.get_all_params(self.l_out)
        self._updates = update_fn(self._loss, model_parameters, learning_rate=self.params.learning_rate)

        self.apply_loss_and_update = theano.function([self.var_K, self.var_X, self.var_Y], self._loss, updates=self._updates)
        self.apply_loss = theano.function([self.var_K, self.var_X, self.var_Y], self._loss)

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

            valid_loss = self.validation_step(X, Y, valid_indices)

            print "Epoch %d training error: %.6f" % (epoch, train_loss)
            print "Epoch %d validation error: %.6f" % (epoch, valid_loss)

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


class DeepNodeClassificationDCNN(NodeClassificationDCNN):
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
        updates = update_fn(loss, model_parameters, learning_rate=self.params.learning_rate)

        self.apply_loss_and_update = theano.function([self.var_A, self.var_X, self.var_Y], loss, updates=updates)
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

            valid_loss = 0.0
            for index in valid_indices:
                valid_loss = self.validation_step(A[index], X[index], Y[index])

            print "Epoch %d training error: %.6f" % (epoch, train_loss)
            print "Epoch %d validation error: %.6f" % (epoch, valid_loss)

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





