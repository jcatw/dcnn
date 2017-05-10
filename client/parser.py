import argparse

import run

from python import params

parser = argparse.ArgumentParser(description="")

parser.add_argument('--model', type=str, default='node_classification')

parser.add_argument('--data', type=str, default='cora')

parser.add_argument('--num_hops', type=int, default=3)
parser.add_argument('--diffusion_threshold', type=float, default=0.05)

parser.add_argument('--learning_rate', type=float, default=0.05)
parser.add_argument('--num_epochs', type=int, default=10)

parser.add_argument('--optimizer', type=str, default='adagrad')

parser.add_argument('--dcnn_nonlinearity', type=str, default='tanh')
parser.add_argument('--dense_nonlinearity', type=str, default='tanh')
parser.add_argument('--out_nonlinearity', type=str, default='softmax')

parser.add_argument('--loss_fn', type=str, default='multiclass_hinge_loss')

parser.add_argument('--stop_window_size', type=int, default=5)
parser.add_argument('--stop_early', type=int, default=1)

parser.add_argument('--num_dcnn_layers', type=int, default=1)
parser.add_argument('--num_dense_layers', type=int, default=1)

parser.add_argument('--dense_layer_size', type=int, default=100)

parser.add_argument('--batch_size', type=int, default=10)

parser.add_argument('--print_train_accuracy', type=int, default=0)
parser.add_argument('--print_valid_accuracy', type=int, default=0)

parser.add_argument('--momentum', type=int, default=0)

parser.add_argument('--explore', type=int, default=0)

parser.add_argument('--check_sparse', type=int, default=0)


def parse():
    args = parser.parse_args()

    parameters = params.Params(
        model=args.model,
        data=args.data,

        num_epochs=args.num_epochs,

        num_hops=args.num_hops,
        diffusion_threshold=args.diffusion_threshold,

        learning_rate=args.learning_rate,

        update_fn=args.optimizer,

        num_nodes=run.metadata[args.data]["num_nodes"],
        num_features=run.metadata[args.data]["num_features"],
        num_classes=run.metadata[args.data]["num_classes"],

        dcnn_nonlinearity=args.dcnn_nonlinearity,
        dense_nonlinearity=args.dense_nonlinearity,
        out_nonlinearity=args.out_nonlinearity,

        loss_fn=args.loss_fn,

        stop_window_size=args.stop_window_size,
        stop_early=args.stop_early,

        batch_size=args.batch_size,
        num_dcnn_layers=args.num_dcnn_layers,
        num_dense_layers=args.num_dense_layers,

        dense_layer_size=args.dense_layer_size,

        print_train_accuracy=args.print_train_accuracy,
        print_valid_accuracy=args.print_valid_accuracy,

        momentum=args.momentum,

        explore=args.explore,
        check_sparse=args.check_sparse,
    )

    return parameters
