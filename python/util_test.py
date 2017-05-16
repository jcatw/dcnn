import unittest

import numpy as np

import data
import util

class TestDiffusionKernelsSynthetic(unittest.TestCase):
    def _parse_data(self):
        self.A = np.asarray([
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ])

    def setUp(self):
        self._parse_data()

        self.num_hops = 3
        self.diffusion_threshold = 0.5
        self.num_nodes = self.A.shape[0]

    def test_A_symmetry(self):
        self.assertTrue((self.A == self.A.T).all())

    def test_num_hops(self):
        self.assertTrue(self.num_hops >= 1)

    def test_num_nodes(self):
        self.assertTrue(self.num_nodes > 0)

    def _skeleton_diffusion_test(self, K):
        # Test identity
        print K[:, 0, :]
        self.assertTrue(K[:, 0, :].sum() == self.num_nodes)

        # Test bounds on kernel
        for i in range(1, self.num_hops + 1):
            print K[:, i, :]
            self.assertTrue(K[:, i, :].min() >= 0.0)
            self.assertTrue(K[:, i, :].max() <= 1.0)

    def test_diffusion_kernel(self):
        K = util.A_to_diffusion_kernel(self.A, self.num_hops)
        self._skeleton_diffusion_test(K)

    def test_pre_thresholded_diffusion_kernel(self):
        K = util.A_to_pre_sparse_diffusion_kernel(self.A, self.num_hops, self.diffusion_threshold)
        self._skeleton_diffusion_test(K)

    def test_post_thresholded_diffusion_kernel(self):
        K = util.A_to_post_sparse_diffusion_kernel(self.A, self.num_hops, self.diffusion_threshold)
        self._skeleton_diffusion_test(K)

class TestDiffusionKernelsCompleteGraph1000(TestDiffusionKernelsSynthetic):
    def _parse_data(self):
        self.A = np.ones((1000, 1000)) - np.eye(1000)


class TestDiffusionKernelsCompleteGraph4(TestDiffusionKernelsSynthetic):
    def _parse_data(self):
        self.A = np.ones((4, 4)) - np.eye(4)


class TestDiffusionKernelsCora(TestDiffusionKernelsSynthetic):
    def _parse_data(self):
        self.A, _, _ = data.parse_cora()

