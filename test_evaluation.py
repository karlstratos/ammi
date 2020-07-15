import evaluation as eva
import torch
import unittest


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.binary_encodings1 = torch.tensor([[1, 1, 1, 1],
                                               [0, 0, 0, 1]])
        self.binary_encodings2 = torch.tensor([[1, 1, 0, 1],
                                               [0, 1, 1, 0],
                                               [0, 0, 0, 0]])

        self.int_encodings1 = torch.tensor([[1, 5, 6, 8],
                                            [9, 10, 3, 5]])
        self.int_encodings2 = torch.tensor([[1, 2, 3, 4],
                                            [5, 6, 7, 8],
                                            [9, 10, 11, 12]])

        self.tgt_label_lists = [['a', 'b'],
                                ['c']]
        self.src_label_lists = [['b', 'c'],
                                ['d'],
                                ['a', 'c']]

    def test_count_correct_matches(self):
        X = torch.tensor([[0, 0], [0, 0], [0, 0]])
        Y = torch.tensor([[0, 0], [0, 0], [0, 0]])
        self.assertEqual(eva.count_correct_matches(X, Y, num_retrieve=3), 3)
        self.assertLess(eva.count_correct_matches(X, Y, num_retrieve=1), 3)

        X = torch.tensor([[0, 0, 0], [1, 1, 1], [1, 0, 1]])
        Y = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1]])

        # D(X,Y) = 1 2 3            -->         x1: [y1, y2, y3]
        #          2 1 0                        x2: [y3, y2, y1]
        #          1 2 1                        x3: [y1==y3, y2]
        self.assertEqual(eva.count_correct_matches(X, Y, num_retrieve=2), 3)
        self.assertEqual(eva.count_correct_matches(X, Y, num_retrieve=1), 1)

    def test_compute_top1_average_precision(self):
        prec1 = eva.compute_topK_average_precision(self.binary_encodings1,
                                                   self.tgt_label_lists,
                                                   self.binary_encodings2,
                                                   self.src_label_lists, 1)
        self.assertEqual(prec1, 100)  # (100 + 100) / 2

    def test_compute_top2_average_precision(self):
        prec2 = eva.compute_topK_average_precision(self.binary_encodings1,
                                                   self.tgt_label_lists,
                                                   self.binary_encodings2,
                                                   self.src_label_lists, 2)
        self.assertEqual(prec2, 75)  # (50 + 100) / 2

    def test_compute_top3_average_precision(self):
        prec3 = eva.compute_topK_average_precision(self.binary_encodings1,
                                                   self.tgt_label_lists,
                                                   self.binary_encodings2,
                                                   self.src_label_lists, 3)
        self.assertAlmostEqual(prec3, 66.67, delta=0.01)  # (66.67 + 66.67) / 2

    def test_compute_hamming_distance_binary(self):
        # binary=False is also okay, only computationally more expensive.
        D = eva.compute_hamming_distance(self.binary_encodings1,
                                         self.binary_encodings2)
        self.assertListEqual(D.tolist(), [[1, 2, 4],
                                          [2, 3, 1]])

    def test_compute_hamming_distance_int(self):
        D = eva.compute_hamming_distance(self.int_encodings1,
                                         self.int_encodings2, binary=False)
        self.assertListEqual(D.tolist(), [[3, 3, 4],
                                          [3, 4, 2]])


if __name__ == '__main__':
    unittest.main()
