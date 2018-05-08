
import unittest
from random import shuffle

import numpy as np
from sklearn.metrics import adjusted_rand_score as ari

from clustermatch.cluster import run_kde_clustering, run_maxdiff_clustering


class MaxDiffClusteringTest(unittest.TestCase):
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
        part = run_maxdiff_clustering(data, 2)

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
        part = run_maxdiff_clustering(data, 2)

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
        part = run_maxdiff_clustering(data, 3)

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
        part = run_maxdiff_clustering(data, 3)

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
        part = run_maxdiff_clustering(data, 4)

        # Validate
        assert part is not None
        assert len(part) == 20
        assert len(np.unique(part)) == 4
        assert ari(data_ref, part) == 1.0

    def test_fixed_points_5_clusters(self):
        # Prepare
        np.random.seed(0)

        data = np.concatenate((
            np.array([-11.5, -11.45, -10.5]), # whole diff: 1.0 -> 0.05, 0.95
            # diff between: 3.0
            np.array([-7.5, -6.5, -6.0]), # whole diff: 1.5 -> 0.50, 1.00
            # diff between: 2.0
            np.array([-4.0, -3.9, -3.5, -3.3, -3.2]), # whole diff: 0.8 -> 0.10, 0.40
            # diff between: 2.5
            np.array([-0.7, -0.6, -0.4, -0.35, 0.5]), # whole diff: 1.2 -> 0.05, 0.85
            # diff between: 5.0
            np.array([5.5, 5.9, 6.3, 6.8, 7.2]), # whole diff: 1.7 -> 0.4, 0.5
        ))
        data_ref = np.concatenate(([0] * 3, [1] * 3, [2] * 5, [3] * 5, [4] * 5))
        assert len(data) == len(data_ref)

        idx_shuffled = list(range(len(data)))
        shuffle(idx_shuffled)

        data = data[idx_shuffled]
        data_ref = data_ref[idx_shuffled]

        # Run
        part = run_maxdiff_clustering(data, 5)

        # Validate
        assert part is not None
        assert len(part) == 21
        assert len(np.unique(part)) == 5
        assert ari(data_ref, part) == 1.0

    def test_fixed_points_6_clusters(self):
        # Prepare
        np.random.seed(0)

        # same data, but rearranged to show more clusters
        data = np.concatenate((
            np.array([-11.5, -11.45, -10.5]), # whole diff: 1.0 -> 0.05, 0.95
            # diff between: 3.0
            # NEXT SPLIT np.array([-7.5, -6.5, -6.0]), # whole diff: 1.5 -> 0.50, 1.00
            np.array([-7.5]), # whole diff: 0.0 -> -
            ## diff between: 1.0
            np.array([-6.5, -6.0]), # whole diff: 0.5 -> -
            # diff between: 2.0
            np.array([-4.0, -3.9, -3.5, -3.3, -3.2]), # whole diff: 0.8 -> 0.10, 0.40
            # diff between: 2.5
            np.array([-0.7, -0.6, -0.4, -0.35, 0.5]), # whole diff: 1.2 -> 0.05, 0.85
            # diff between: 5.0
            np.array([5.5, 5.9, 6.3, 6.8, 7.2]), # whole diff: 1.7 -> 0.4, 0.5
        ))
        data_ref = np.concatenate(([0] * 3, [5] * 1, [6] * 2, [2] * 5, [3] * 5, [4] * 5))
        assert len(data) == len(data_ref)

        idx_shuffled = list(range(len(data)))
        shuffle(idx_shuffled)

        data = data[idx_shuffled]
        data_ref = data_ref[idx_shuffled]

        # Run
        part = run_maxdiff_clustering(data, 6)

        # Validate
        assert part is not None
        assert len(part) == 21
        assert len(np.unique(part)) == 6
        assert ari(data_ref, part) == 1.0

    def test_fixed_points_10_clusters(self):
        # Prepare
        np.random.seed(0)

        # same data, but rearranged to show more clusters
        data = np.concatenate((
            np.array([-11.5, -11.45, -10.5]), # whole diff: 1.0 -> 0.05, 0.95
            # diff between: 3.0
            # NEXT SPLIT np.array([-7.5, -6.5, -6.0]), # whole diff: 1.5 -> 0.50, 1.00
            np.array([-7.5]), # whole diff: 0.0 -> -
            ## diff between: 1.0
            np.array([-6.5, -6.0]), # whole diff: 0.5 -> -
            # diff between: 2.0
            np.array([-4.0, -3.9, -3.5, -3.3, -3.2]), # whole diff: 0.8 -> 0.10, 0.40
            # diff between: 2.5
            np.array([-0.7, -0.6, -0.4, -0.35, 0.5]), # whole diff: 1.2 -> 0.05, 0.85
            # diff between: 5.0
            np.array([5.5, 5.9, 6.3, 6.8, 7.2]), # whole diff: 1.7 -> 0.4, 0.5
        ))
        data_ref = np.concatenate(([0] * 3, [5] * 1, [6] * 2, [2] * 5, [3] * 5, [4] * 5))
        assert len(data) == len(data_ref)

        idx_shuffled = list(range(len(data)))
        shuffle(idx_shuffled)

        data = data[idx_shuffled]
        data_ref = data_ref[idx_shuffled]

        # Run
        part = run_maxdiff_clustering(data, 6)

        # Validate
        assert part is not None
        assert len(part) == 21
        assert len(np.unique(part)) == 6
        assert ari(data_ref, part) == 1.0