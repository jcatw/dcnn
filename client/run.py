import numpy as np
from sklearn import metrics
import random

import parser

from python import data
from python import models
from python import params as params_mod


metadata = {
    "cora": {
        "parser": data.parse_cora,
        "num_nodes": 2708,
        "num_features": 1433,
        "num_classes": 7,
    },
    'nci1': {
        "parser": lambda: data.parse_graph_data('nci1.graph'),
        "num_nodes": None,
        "num_graphs": None,
        "num_features": 37,
        "num_classes": 2,
    }

}

model_map = {
    "node_classification": models.NodeClassificationDCNN,
    "deep_node_classification": models.DeepNodeClassificationDCNN,
    "deep_dense_node_classification": models.DeepDenseNodeClassificationDCNN,
    "graph_classification": models.GraphClassificationDCNN,
    "deep_graph_classification": models.DeepGraphClassificationDCNN,
    "feature_aggregated_graph_classification": models.GraphClassificationFeatureAggregatedDCNN,
}

hyperparameter_choices = {
    "num_hops": range(6),
    "learning_rate": [0.01, 0.05, 0.1, 0.25],
    "optimizer": params_mod.update_map.keys(),
    "loss": params_mod.loss_map.keys(),
    "dcnn_nonlinearity": params_mod.nonlinearity_map.keys(),
    "dense_nonlinearity": params_mod.nonlinearity_map.keys(),
    "num_epochs": [10, 100, 1000],
    "batch_size": [10, 100],
    "early_stopping": [0, 1],
    "stop_window_size": [1, 5, 10],
    "num_dcnn_layers": range(1, 6),
    "num_dense_layers": range(1, 6),
}

def run_node_classification(parameters):
    A, X, Y = metadata[parameters.data]["parser"]()

    dcnn = model_map[parameters.model](parameters, A)

    num_nodes = A.shape[0]

    indices = np.arange(num_nodes).astype('int32')
    np.random.shuffle(indices)

    train_indices = indices[:num_nodes // 3]
    valid_indices = indices[num_nodes // 3: (2 * num_nodes) // 3]
    test_indices = indices[(2 * num_nodes) // 3:]

    dcnn.fit(X, Y, train_indices, valid_indices)

    predictions = dcnn.predict(X, test_indices)
    actuals = Y[test_indices, :].argmax(1)

    accuracy = metrics.accuracy_score(actuals, predictions)

    print "Test Accuracy: %.4f" % (accuracy,)

def run_graph_classification(params):
    print "parsing data..."
    A, X, Y = metadata[parameters.data]["parser"]()

    # Shuffle the data.
    tmp = list(zip(A, X, Y))
    random.shuffle(tmp)
    A, X, Y = zip(*tmp)

    num_graphs = len(A)

    indices = np.arange(num_graphs).astype('int32')
    np.random.shuffle(indices)

    train_indices = indices[:num_graphs // 3]
    valid_indices = indices[num_graphs // 3: (2 * num_graphs) // 3]
    test_indices = indices[(2 * num_graphs) // 3:]

    print "initializing model..."
    m = model_map[params.model](params)

    print "training model..."
    m.fit(A, X, Y, train_indices=train_indices, valid_indices=valid_indices)

    test_predictions = []
    test_actuals = []
    for test_index in test_indices:
        pred = m.predict(A[test_index], X[test_index])
        test_predictions.append(pred)
        test_actuals.append(Y[test_index].argmax())

    test_accuracy = 0.0
    num_test = len(test_predictions)
    for i in range(len(test_predictions)):
        if (test_predictions[i] == test_actuals[i]):
            test_accuracy += 1.0

    test_accuracy /= num_test
    print "Test Accuracy: %.6f" % (test_accuracy,)
    print params
    print "RESULTS:%.6f" % test_accuracy
    print "done"

if __name__ == '__main__':
    parameters = parser.parse()

    if parameters.explore:
        while True:
            for possibility in hyperparameter_choices.keys():
                choice = random.choice(hyperparameter_choices[possibility])
                parameters.set(possibility, choice)

            print "Parameter choices:"
            print parameters

            if "node_classification" in parameters.model:
                run_node_classification(parameters)
            else:
                run_graph_classification(parameters)

    if "node_classification" in parameters.model:
        run_node_classification(parameters)
    else:
        run_graph_classification(parameters)