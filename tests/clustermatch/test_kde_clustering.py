
import unittest
from random import shuffle

import numpy as np
from sklearn.metrics import adjusted_rand_score as ari

from clustermatch.cluster import run_kde_clustering, run_maxdiff_clustering


class KDEClusteringTest(unittest.TestCase):
    def test_two_clusters01(self):
        # Prepare
        np.random.seed(0)

        data = np.concatenate((
            np.random.normal(0, 1, 13),
            np.random.normal(5, 1, 7)
        ))
        data_ref = np.concatenate(([0] * 13, [1] * 7))

        idx_shuffled = list(range(len(data)))
        shuffle(idx_shuffled)

        data = data[idx_shuffled]
        data_ref = data_ref[idx_shuffled]

        # Run
        part = run_kde_clustering(data, 2)

        # Validate
        assert part is not None
        assert len(part) == 20
        assert len(np.unique(part)) == 2
        assert ari(data_ref, part) == 1.0

    def test_two_clusters02(self):
        # Prepare
        np.random.seed(0)

        data = np.concatenate((
            np.random.normal(-3, 0.5, 7),
            np.random.normal(0, 1, 7),
            np.random.normal(5, 1, 6)
        ))
        data_ref = np.concatenate(([0] * 14, [1] * 6))

        idx_shuffled = list(range(len(data)))
        shuffle(idx_shuffled)

        data = data[idx_shuffled]
        data_ref = data_ref[idx_shuffled]

        # Run
        part = run_kde_clustering(data, 2)

        # Validate
        assert part is not None
        assert len(part) == 20
        assert len(np.unique(part)) == 2
        assert ari(data_ref, part) == 1.0

    def test_three_clusters01(self):
        # Prepare
        np.random.seed(0)

        data = np.concatenate((
            np.random.normal(-3, 0.5, 7),
            np.random.normal(0, 1, 7),
            np.random.normal(5, 1, 6)
        ))
        data_ref = np.concatenate(([0] * 7, [1] * 7, [2] * 6))

        idx_shuffled = list(range(len(data)))
        shuffle(idx_shuffled)

        data = data[idx_shuffled]
        data_ref = data_ref[idx_shuffled]

        # Run
        part = run_kde_clustering(data, 3)

        # Validate
        assert part is not None
        assert len(part) == 20
        assert len(np.unique(part)) == 3
        assert ari(data_ref, part) == 1.0

    def test_three_clusters02(self):
        # Prepare
        np.random.seed(0)

        data = np.concatenate((
            np.random.normal(-5, 1, 6),
            np.random.normal(0, 1, 7),
            np.random.normal(4, 0.5, 7),
        ))
        data_ref = np.concatenate(([0] * 6, [1] * 7, [2] * 7))

        idx_shuffled = list(range(len(data)))
        shuffle(idx_shuffled)

        data = data[idx_shuffled]
        data_ref = data_ref[idx_shuffled]

        # Run
        part = run_kde_clustering(data, 3)

        # Validate
        assert part is not None
        assert len(part) == 20
        assert len(np.unique(part)) == 3
        assert ari(data_ref, part) == 1.0

    def test_four_clusters01(self):
        # Prepare
        np.random.seed(0)

        data = np.concatenate((
            np.random.normal(-5, 1, 5),
            np.random.normal(0, 0.5, 5),
            np.random.normal(3, 0.5, 5),
            np.random.normal(7, 1, 5),
        ))
        data_ref = np.concatenate(([0] * 5, [1] * 5, [2] * 5, [3] * 5))

        idx_shuffled = list(range(len(data)))
        shuffle(idx_shuffled)

        data = data[idx_shuffled]
        data_ref = data_ref[idx_shuffled]

        # Run
        part = run_kde_clustering(data, 4)

        # Validate
        assert part is not None
        assert len(part) == 20
        assert len(np.unique(part)) == 4
        assert ari(data_ref, part) == 1.0
