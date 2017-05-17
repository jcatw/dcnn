import unittest

import data

class TestCora(unittest.TestCase):
    def setUp(self):
        self.A, self.X, self.Y = data.parse_cora()

    def test_A_symmetry(self):
        self.assertTrue( (self.A == self.A.T).all() )

    def test_A_min(self):
        self.assertTrue( self.A.min() >= 0.0 )

    def test_A_max(self):
        self.assertTrue( self.A.max() <= 1.0 )

    def test_max_degree(self):
        self.assertTrue(self.A.sum(0).max() == 168.0)
        self.assertTrue(self.A.sum(1).max() == 168.0)

    def test_no_self_loops(self):
        self_loops = False
        for i in range(self.A.shape[0]):
            self_loops = self_loops or self.A[i,i] != 0.0

        self.assertFalse(self_loops)


class TestSparseCora(TestCora):
    def setUp(self):
        self.A, self.X, self.Y = data.parse_cora_sparse()

    def test_A_symmetry(self):
        # Not sure how to do this
        self.assertTrue(False)

