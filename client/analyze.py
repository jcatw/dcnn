import parser
import run

from python import util

def check_sparse_transition(A, parameters, thresholded=False):
    P = A / (A.sum(0) + 1.0)

    if thresholded:
        P[P <= parameters.diffusion_threshold] = 0
        P[P > parameters.diffusion_threshold] = 1
    else:
        P[P != 0] = 1

    non_zeros = P.sum()
    num_nodes = A.shape[1]
    num_entries = num_nodes**2

    print "num nodes: %d" % num_nodes
    print "num entries: %d" % num_entries
    print "non zeros: %d" % non_zeros
    print "occupied proportion: %.6f" % (float(non_zeros) / num_entries,)
    print ""

def check_sparse_kernel(A, parameters, post=False):
    if post:
        K = util.A_to_post_sparse_diffusion_kernel(A, parameters.num_hops, parameters.diffusion_threshold)
    else:
        K = util.A_to_pre_sparse_diffusion_kernel(A, parameters.num_hops, parameters.diffusion_threshold)

    K[K != 0] = 1

    non_zeros = K.sum()
    num_nodes = A.shape[1]

    num_entries = (parameters.num_hops + 1) * (num_nodes ** 2)

    print "num nodes: %d" % num_nodes
    print "num entries: %d" % num_entries
    print "non zeros: %d" % non_zeros
    print "occupied proportion: %.6f" % (float(non_zeros) / num_entries,)
    print ""


if __name__ == '__main__':
    parameters = parser.parse()

    if parameters.check_sparse:
        A, X, Y = run.metadata[parameters.data]["parser"]()

        print "Checking transition sparseness..."
        check_sparse_transition(A, parameters)

        print "Checking thresholded transition sparseness..."
        check_sparse_transition(A, parameters, thresholded=True)

        print "Checking pre-thresholded kernel sparseness..."
        check_sparse_kernel(A, parameters)

        print "Checking post-thresholded kernel sparseness..."
        check_sparse_kernel(A, parameters, post=True)
