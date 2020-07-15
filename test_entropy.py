# Author: Karl Stratos (karlstratos@gmail.com)

import entropy as ent
import itertools
import math
import numpy as np
import torch
import unittest


class TestEntropy(unittest.TestCase):

    def setUp(self):
        self.Bs = [1, 100]      # Batch sizes
        self.ms = [1, 5]        # Numbers of binary variables
        self.os = [0, 1, 3]     # Markov orders of p(z|y)
        self.rs = [0, 1, 3, 5]  # Markov orders of q(z|x)

    def sample_model(self, B, m, o):
        weights_P = torch.randn(B, m, pow(2, o))
        P_ = torch.cat([-weights_P, weights_P], dim=2)
        P = torch.sigmoid(P_)
        return P, P_

    def test_estimate_hZ_X(self):
        for B, m, o, r in itertools.product(self.Bs, self.ms, self.os, self.rs):
            if r >= o:
                P, _ = self.sample_model(B, m, o)
                Q, Q_ = self.sample_model(B, m, r)
                hZ_X = ent.estimate_hZ_X(P, Q_)
                hZ_X_brute = ent.estimate_hZ_X_brute(P, Q)
                self.assertAlmostEqual(hZ_X, hZ_X_brute, delta=0.00005)

    def test_compute_viterbi(self):
        for B, m, o in itertools.product(self.Bs, self.ms, self.os):
            P, _ = self.sample_model(B, m, o)
            zs, max_probs = ent.compute_viterbi(P)
            zs_brute, max_probs_brute = ent.compute_viterbi_brute(P)
            self.assertListEqual(zs.tolist(), zs_brute.tolist())
            for b in range(B):
                self.assertAlmostEqual(max_probs[b], max_probs_brute[b],
                                       delta=0.00005)

    def test_estimate_hZ(self):
        for B, m, o in itertools.product(self.Bs, self.ms, self.os):
            if o <= m:
                P, _ = self.sample_model(B, m, o)
                hZ = ent.estimate_hZ(P)
                hZ_brute = ent.estimate_hZ_brute(P)
                self.assertAlmostEqual(hZ, hZ_brute, delta=0.00005)

    def test_compute_prob(self):
        P = torch.FloatTensor(
            [[[0.7306, 0.0067, 0.2200, 0.9348, 0.2694, 0.9933, 0.7800, 0.0652],
              [0.7605, 0.0401, 0.9690, 0.4395, 0.2395, 0.9599, 0.0310, 0.5605],
              [0.7370, 0.5805, 0.1471, 0.9961, 0.2630, 0.4195, 0.8529, 0.0039],
              [0.8812, 0.4859, 0.4366, 0.2164, 0.1188, 0.5141, 0.5634, 0.7836]]]
        )
        prob = ent.compute_prob(P, [1, 0, 1, 1])[0]
        self.assertAlmostEqual(prob, 0.0617, delta=0.00005)

    def test_precompute_quads(self):
        quads = ent.precompute_quads(2)
        self.assertListEqual(quads.tolist(), [[0, 0, 1, 1], [2, 2, 3, 3],
                                              [0, 4, 1, 5], [2, 6, 3, 7]])

    def test_precompute_I(self):
        I = ent.precompute_I(3, 2).tolist()

        # [[0, 1, 0, 1, 0, 1, 0, 1],      [[0, 4, 0, 4, 0, 4, 0, 4],
        #  [0, 0, 1, 1, 0, 0, 1, 1],  =>   [0, 2, 4, 6, 0, 2, 4, 6],
        #  [0, 0, 0, 0, 1, 1, 1, 1]]       [0, 1, 2, 3, 4, 5, 6, 7]]
        self.assertListEqual(I, [[0, 4, 0, 4, 0, 4, 0, 4],
                                 [0, 2, 4, 6, 0, 2, 4, 6],
                                 [0, 1, 2, 3, 4, 5, 6, 7]])


if __name__ == '__main__':
    unittest.main()
