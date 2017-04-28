import numpy as np

def A_to_diffusion_kernel(A, k):
    """
    Computes [A**0, A**1, ..., A**k]

    :param A: 2d numpy array
    :param k: integer, degree of series
    :return: 3d numpy array [A**0, A**1, ..., A**k]
    """
    assert k >= 0

    Apow = [np.identity(A.shape[0])]

    if k > 0:
        d = A / A.sum(0)

        Apow.append(A / (d + 1.0))

        for i in range(2, k+1):
            Apow.append(np.dot(A / (d + 1.0), Apow[-1]))

    return np.transpose(np.asarray(Apow, dtype='float32'), (1, 0, 2))

def A_to_post_sparse_diffusion_kernel(A, k, threshold):
    K = A_to_diffusion_kernel(A, k)

    K[K < threshold] = 0.0

    return K