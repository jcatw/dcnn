import cPickle as cp
import inspect
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

def parse_cora():
    path = "%s/../data/cora/" % (current_dir,)

    id2index = {}

    label2index = {
        'Case_Based': 0,
        'Genetic_Algorithms': 1,
        'Neural_Networks': 2,
        'Probabilistic_Methods': 3,
        'Reinforcement_Learning': 4,
        'Rule_Learning': 5,
        'Theory': 6
    }

    features = []
    labels = []

    with open(path + 'cora.content', 'r') as f:
        i = 0
        for line in f.xreadlines():
            items = line.strip().split('\t')

            id = items[0]

            # 1-hot encode labels
            label = np.zeros(len(label2index))
            label[label2index[items[-1]]] = 1
            labels.append(label)

            # parse features
            features.append([int(x) for x in items[1:-1]])

            id2index[id] = i
            i += 1

    features = np.asarray(features, dtype='float32')
    labels = np.asarray(labels, dtype='int32')

    n_papers = len(id2index)

    adj = np.zeros((n_papers, n_papers), dtype='float32')

    with open(path + 'cora.cites', 'r') as f:
        for line in f.xreadlines():
            items = line.strip().split('\t')
            adj[id2index[items[0]], id2index[items[1]]] = 1.0
            # undirected
            adj[id2index[items[1]], id2index[items[0]]] = 1.0

    return adj.astype('float32'), features.astype('float32'), labels.astype('int32')

def parse_graph_data(graph_name='nci1.graph'):
    path = "%s/../data/" % (current_dir,)

    if graph_name == 'nci1.graph':
        maxval = 37
        n_classes = 2
    elif graph_name == 'nci109.graph':
        maxval = 38
        n_classes = 2
    elif graph_name == 'mutag.graph':
        maxval = 7
        n_classes = 2
    elif graph_name == 'ptc.graph':
        maxval = 22
        n_classes = 2
    elif graph_name == 'enzymes.graph':
        maxval = 3
        n_classes = 6

    with open(path+graph_name,'r') as f:
        raw = cp.load(f)

        n_graphs = len(raw['graph'])

        A = []
        rX = []
        Y = []

        for i in range(n_graphs):
            # Set label
            class_label = raw['labels'][i]

            y = np.zeros((1, n_classes), dtype='int32')

            if n_classes == 2:
                if class_label == 1:
                    y[0,1] = 1
                else:
                    y[0,0] = 1
            else:
                y[0,class_label-1] = 1

            # Parse graph
            G = raw['graph'][i]

            n_nodes = len(G)

            a = np.zeros((n_nodes, n_nodes), dtype='float32')
            x = np.zeros((n_nodes, maxval), dtype='float32')

            for node, meta in G.iteritems():
                label = meta['label'][0] - 1
                x[node, label] = 1
                for neighbor in meta['neighbors']:
                    a[node, neighbor] = 1

            A.append(a)
            rX.append(x)
            Y.append(y)

    return A, rX, Y
