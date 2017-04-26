import numpy as np
from sklearn import metrics
import random

import parser

from python import data
from python import models


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
    "feature_aggregated_graph_classification": models.GraphClassificationFeatureAggregatedDCNN,
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
    print "done"

if __name__ == '__main__':
    parameters = parser.parse()

    if "node_classification" in parameters.model:
        run_node_classification(parameters)
    else:
        run_graph_classification(parameters)