# coding=utf-8

import unittest
from itertools import combinations, product
from time import time

import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import adjusted_rand_score as ari

from clustermatch.cluster import calculate_simmatrix, get_squareform, run_quantile_clustering
from tests.utils import get_data_file
from utils.data import merge_sources


class ClusterMatchSimilarityMatrixTest(unittest.TestCase):
    def test_sim_matrix_shape(self):
        ## Prepare
        # 10/7 measures and 50 tomatoes/columns
        source01 = np.random.rand(10, 20)
        source02 = np.random.rand(7, 20)

        data = np.concatenate((source01, source02))

        ## Run
        simmatrix = calculate_simmatrix(data, n_jobs=1)

        ## Validate
        assert simmatrix is not None
        assert hasattr(simmatrix, 'shape')
        assert len(simmatrix.shape) == 1
        assert simmatrix.shape[0] == 136
        assert squareform(simmatrix).shape == (17, 17)

    def test_artificial_data_complete(self):
        ## Prepare
        np.random.seed(0)

        # source01
        source01 = np.array([
            [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  #00 1 <<
            [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  #01 1 <<
            [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  #02 1 <<
            [0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],  #03 2 --
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #04 2 --
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  #05 3
        ], dtype=np.float)
        # source01 = source01 + np.random.rand(6, 7) * 0.2

        # source02
        source02 = np.array([
            [100, 100,   0,    0,   0,  0,    0.0],  #06 2
            [ 50,  50,   0,    0,   0,  0,    0.0],  #07 2 --
            [  0,   0,   0,    0,   0,  0,    0.1],  #08 3
            [900, 900,   0,    0,   0,  0,    0.0],  #09 2 --
            [  0,   0,  50,  100,  50,  0,    0.0],  #10 1 <<
            [  0,   0,   0,    0,   0,  0, 1000.0],  #11 3
            [  0,   0, 500, 1000, 500,  0,    0.0],  #12 1 <<
            [100, 100,   0,    0,   0,  0,    0.0],  #13 2 --
            [  0,   0,   0,    0,   0,  0,  100.0],  #14 3
        ], dtype=np.float)
        # source02 = source02 + np.random.rand(9, 7) * 20

        data = np.concatenate((source01, source02))

        ## Run
        simmatrix = calculate_simmatrix(data, internal_n_clusters=2, n_jobs=1)

        ## Validate
        assert simmatrix is not None
        assert hasattr(simmatrix, 'shape')
        assert len(simmatrix.shape) == 1
        assert simmatrix.shape[0] == 105

        simmatrix_square = get_squareform(simmatrix)
        assert simmatrix_square.shape == (15, 15)
        assert np.all(np.diag(simmatrix_square) == 1.0)

        cluster01_members = (0, 1, 2, 10, 12)
        cluster02_members = (3, 4, 6, 7, 9, 13)
        cluster03_members = (5, 8, 11, 14)

        assert len(np.unique(cluster01_members)) + \
               len(np.unique(cluster02_members)) + \
               len(np.unique(cluster03_members)) == 15

        cluster01_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster01_members, 2)])
        assert np.all(cluster01_sim == 1.0), cluster01_sim

        cluster02_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster02_members, 2)])
        assert np.all(cluster02_sim == 1.0), cluster02_sim

        cluster03_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster03_members, 2)])
        assert np.all(cluster03_sim == 1.0), cluster03_sim

        cluster01_against_cluster02 = np.array([simmatrix_square[idx] for idx in product(cluster01_members, cluster02_members)])
        assert np.all(cluster01_against_cluster02 <= 0.50), cluster01_against_cluster02

        cluster01_against_cluster03 = np.array([simmatrix_square[idx] for idx in product(cluster01_members, cluster03_members)])
        assert np.all(cluster01_against_cluster03 <= 0.50), cluster01_against_cluster03

        cluster02_against_cluster03 = np.array([simmatrix_square[idx] for idx in product(cluster02_members, cluster03_members)])
        assert np.all(cluster02_against_cluster03 <= 0.50), cluster02_against_cluster03

    def test_artificial_data_column_with_some_nan(self):
        ## Prepare
        np.random.seed(0)

        # source01
        source01 = np.array([
            [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  #00 1 <<
            [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  #01 1 <<
            [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  #02 1 <<
            [0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],  #03 2 --
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #04 2 --
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  #05 3
        ], dtype=np.float)
        # source01 = source01 + np.random.rand(6, 7) * 0.2

        # source02
        source02 = np.array([
            [100, 100,   0,    0,   0,  np.nan,    0.0],  #06 2
            [ 50,  50,   0,    0,   0,       0,    0.0],  #07 2 --
            [  0,   0,   0,    0,   0,  np.nan,    0.1],  #08 3
            [900, 900,   0,    0,   0,  np.nan,    0.0],  #09 2 --
            [  0,   0,  50,  100,  50,       0,    0.0],  #10 1 <<
            [  0,   0,   0,    0,   0,  np.nan, 1000.0],  #11 3
            [  0,   0, 500, 1000, 500,  np.nan,    0.0],  #12 1 <<
            [100, 100,   0,    0,   0,       0,    0.0],  #13 2 --
            [  0,   0,   0,    0,   0,  np.nan,  100.0],  #14 3
        ], dtype=np.float)
        # source02 = source02 + np.random.rand(9, 7) * 20

        data = np.concatenate((source01, source02))

        ## Run
        simmatrix = calculate_simmatrix(data, internal_n_clusters=2, n_jobs=1)

        ## Validate
        assert simmatrix is not None
        assert hasattr(simmatrix, 'shape')
        assert len(simmatrix.shape) == 1
        assert simmatrix.shape[0] == 105

        simmatrix_square = get_squareform(simmatrix)
        assert simmatrix_square.shape == (15, 15)
        assert np.all(np.diag(simmatrix_square) == 1.0)

        cluster01_members = (0, 1, 2, 10, 12)
        cluster02_members = (3, 4, 6, 7, 9, 13)
        cluster03_members = (5, 8, 11, 14)

        assert len(np.unique(cluster01_members)) + \
               len(np.unique(cluster02_members)) + \
               len(np.unique(cluster03_members)) == 15

        cluster01_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster01_members, 2)])
        assert np.all(cluster01_sim == 1.0), cluster01_sim

        cluster02_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster02_members, 2)])
        assert np.all(cluster02_sim == 1.0), cluster02_sim

        cluster03_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster03_members, 2)])
        assert np.all(cluster03_sim == 1.0), cluster03_sim

        # in this case, missing elements makes some rows of cluster01 more similar to some cluster02 members
        cluster01_against_cluster02 = np.array([simmatrix_square[idx] for idx in product(cluster01_members, cluster02_members)])
        assert np.all(cluster01_against_cluster02 <= 0.50), cluster01_against_cluster02

        cluster01_against_cluster03 = np.array([simmatrix_square[idx] for idx in product(cluster01_members, cluster03_members)])
        assert np.all(cluster01_against_cluster03 <= 0.50), cluster01_against_cluster03

        cluster02_against_cluster03 = np.array([simmatrix_square[idx] for idx in product(cluster02_members, cluster03_members)])
        assert np.all(cluster02_against_cluster03 <= 0.50), cluster02_against_cluster03

    def test_artificial_data_column_all_nan(self):
        ## Prepare
        np.random.seed(0)

        # source01
        source01 = np.array([
            [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  #00 1 <<
            [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  #01 1 <<
            [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  #02 1 <<
            [0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],  #03 2 --
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #04 2 --
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  #05 3
        ], dtype=np.float)
        # source01 = source01 + np.random.rand(6, 7) * 0.2

        # source02
        source02 = np.array([
            [100, 100,   0,    0,   0,  np.nan,    0.0],  #06 2
            [ 50,  50,   0,    0,   0,  np.nan,    0.0],  #07 2 --
            [  0,   0,   0,    0,   0,  np.nan,    0.1],  #08 3
            [900, 900,   0,    0,   0,  np.nan,    0.0],  #09 2 --
            [  0,   0,  50,  100,  50,  np.nan,    0.0],  #10 1 <<
            [  0,   0,   0,    0,   0,  np.nan, 1000.0],  #11 3
            [  0,   0, 500, 1000, 500,  np.nan,    0.0],  #12 1 <<
            [100, 100,   0,    0,   0,  np.nan,    0.0],  #13 2 --
            [  0,   0,   0,    0,   0,  np.nan,  100.0],  #14 3
        ], dtype=np.float)
        # source02 = source02 + np.random.rand(9, 7) * 20

        data = np.concatenate((source01, source02))

        ## Run
        simmatrix = calculate_simmatrix(data, internal_n_clusters=2, n_jobs=1)

        ## Validate
        assert simmatrix is not None
        assert hasattr(simmatrix, 'shape')
        assert len(simmatrix.shape) == 1
        assert simmatrix.shape[0] == 105

        simmatrix_square = get_squareform(simmatrix)
        assert simmatrix_square.shape == (15, 15)
        assert np.all(np.diag(simmatrix_square) == 1.0)

        cluster01_members = (0, 1, 2, 10, 12)
        cluster02_members = (3, 4, 6, 7, 9, 13)
        cluster03_members = (5, 8, 11, 14)

        assert len(np.unique(cluster01_members)) + \
               len(np.unique(cluster02_members)) + \
               len(np.unique(cluster03_members)) == 15

        cluster01_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster01_members, 2)])
        assert np.all(cluster01_sim == 1.0), cluster01_sim

        cluster02_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster02_members, 2)])
        assert np.all(cluster02_sim == 1.0), cluster02_sim

        cluster03_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster03_members, 2)])
        assert np.all(cluster03_sim == 1.0), cluster03_sim

        # in this case, missing elements makes some rows of cluster01 more similar to some cluster02 members
        cluster01_against_cluster02 = np.array([simmatrix_square[idx] for idx in product(cluster01_members, cluster02_members)])
        assert np.all(cluster01_against_cluster02 <= 0.50), cluster01_against_cluster02

        cluster01_against_cluster03 = np.array([simmatrix_square[idx] for idx in product(cluster01_members, cluster03_members)])
        assert np.all(cluster01_against_cluster03 <= 0.50), cluster01_against_cluster03

        cluster02_against_cluster03 = np.array([simmatrix_square[idx] for idx in product(cluster02_members, cluster03_members)])
        assert np.all(cluster02_against_cluster03 <= 0.50), cluster02_against_cluster03

    def test_artificial_data_random_nan_values_in_one_source(self):
        ## Prepare
        np.random.seed(0)

        # source01
        source01 = np.array([
            [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  #00 1 <<
            [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  #01 1 <<
            [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  #02 1 <<
            [0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],  #03 2 --
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #04 2 --
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  #05 3
        ], dtype=np.float)
        # source01 = source01 + np.random.rand(6, 7) * 0.2

        # source02
        source02 = np.array([
            [100,      100,        0,    0,   0,       0,    0.0],  #06 2
            [ 50,       50,   np.nan,    0,   0,       0,    0.0],  #07 2 --
            [  0,        0,        0,    0,   0,       0,    0.1],  #08 3
            [900,      900,        0,    0,   0,  np.nan,    0.0],  #09 2 --
            [  0,        0,       50,  100,  50,       0,    0.0],  #10 1 <<
            [  0,   np.nan,        0,    0,   0,       0, 1000.0],  #11 3
            [  0,        0,      500, 1000, 500,  np.nan,    0.0],  #12 1 <<
            [100,      100,        0,    0,   0,       0,    0.0],  #13 2 --
            [  0,        0,        0,    0,   0,       0,  100.0],  #14 3
        ], dtype=np.float)
        # source02 = source02 + np.random.rand(9, 7) * 20

        data = np.concatenate((source01, source02))

        ## Run
        simmatrix = calculate_simmatrix(data, internal_n_clusters=2, n_jobs=1)

        ## Validate
        assert simmatrix is not None
        assert hasattr(simmatrix, 'shape')
        assert len(simmatrix.shape) == 1
        assert simmatrix.shape[0] == 105

        simmatrix_square = get_squareform(simmatrix)
        assert simmatrix_square.shape == (15, 15)
        assert np.all(np.diag(simmatrix_square) == 1.0)

        cluster01_members = (0, 1, 2, 10, 12)
        cluster02_members = (3, 4, 6, 7, 9, 13)
        cluster03_members = (5, 8, 11, 14)

        assert len(np.unique(cluster01_members)) + \
               len(np.unique(cluster02_members)) + \
               len(np.unique(cluster03_members)) == 15

        cluster01_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster01_members, 2)])
        assert np.all(cluster01_sim == 1.0), cluster01_sim

        cluster02_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster02_members, 2)])
        assert np.all(cluster02_sim == 1.0), cluster02_sim

        cluster03_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster03_members, 2)])
        assert np.all(cluster03_sim == 1.0), cluster03_sim

        # in this case, missing elements makes some rows of cluster01 more similar to some cluster02 members
        cluster01_against_cluster02 = np.array([simmatrix_square[idx] for idx in product(cluster01_members, cluster02_members)])
        assert np.all(cluster01_against_cluster02 <= 0.50), cluster01_against_cluster02

        cluster01_against_cluster03 = np.array([simmatrix_square[idx] for idx in product(cluster01_members, cluster03_members)])
        assert np.all(cluster01_against_cluster03 <= 0.50), cluster01_against_cluster03

        cluster02_against_cluster03 = np.array([simmatrix_square[idx] for idx in product(cluster02_members, cluster03_members)])
        assert np.all(cluster02_against_cluster03 <= 0.50), cluster02_against_cluster03

    def test_artificial_data_random_nan_values_in_all_sources(self):
        ## Prepare
        np.random.seed(0)

        # source01
        source01 = np.array([
            [0.0, 0.0,    0.5, 1.0, 0.5, np.nan, 0.0],  #00 1 <<
            [0.0, 0.0,    0.5, 1.0, 0.5,    0.0, 0.0],  #01 1 <<
            [0.0, 0.0,    0.5, 1.0, 0.5,    0.0, 0.0],  #02 1 <<
            [0.8, 0.8, np.nan, 0.0, 0.0,    0.0, 0.0],  #03 2 --
            [1.0, 1.0,    0.0, 0.0, 0.0,    0.0, 0.0],  #04 2 --
            [0.0, 0.0,    0.0, 0.0, 0.0,    0.0, 1.0],  #05 3
        ], dtype=np.float)
        # source01 = source01 + np.random.rand(6, 7) * 0.2

        # source02
        source02 = np.array([
            [100,      100,        0,    0,   0,       0,    0.0],  #06 2
            [ 50,       50,   np.nan,    0,   0,       0,    0.0],  #07 2 --
            [  0,        0,        0,    0,   0,       0,    0.1],  #08 3
            [900,      900,        0,    0,   0,  np.nan,    0.0],  #09 2 --
            [  0,        0,       50,  100,  50,       0,    0.0],  #10 1 <<
            [  0,   np.nan,        0,    0,   0,       0, 1000.0],  #11 3
            [  0,        0,      500, 1000, 500,  np.nan,    0.0],  #12 1 <<
            [100,      100,        0,    0,   0,       0,    0.0],  #13 2 --
            [  0,        0,        0,    0,   0,       0,  100.0],  #14 3
        ], dtype=np.float)
        # source02 = source02 + np.random.rand(9, 7) * 20

        data = np.concatenate((source01, source02))

        ## Run
        simmatrix = calculate_simmatrix(data, internal_n_clusters=2, n_jobs=1)

        ## Validate
        assert simmatrix is not None
        assert hasattr(simmatrix, 'shape')
        assert len(simmatrix.shape) == 1
        assert simmatrix.shape[0] == 105

        simmatrix_square = get_squareform(simmatrix)
        assert simmatrix_square.shape == (15, 15)
        assert np.all(np.diag(simmatrix_square) == 1.0)

        cluster01_members = (0, 1, 2, 10, 12)
        cluster02_members = (3, 4, 6, 7, 9, 13)
        cluster03_members = (5, 8, 11, 14)

        assert len(np.unique(cluster01_members)) + \
               len(np.unique(cluster02_members)) + \
               len(np.unique(cluster03_members)) == 15

        cluster01_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster01_members, 2)])
        assert np.all(cluster01_sim == 1.0), cluster01_sim

        cluster02_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster02_members, 2)])
        assert np.all(cluster02_sim == 1.0), cluster02_sim

        cluster03_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster03_members, 2)])
        assert np.all(cluster03_sim == 1.0), cluster03_sim

        # in this case, missing elements makes some rows of cluster01 more similar to some cluster02 members
        cluster01_against_cluster02 = np.array([simmatrix_square[idx] for idx in product(cluster01_members, cluster02_members)])
        assert np.all(cluster01_against_cluster02 <= 0.50), cluster01_against_cluster02

        cluster01_against_cluster03 = np.array([simmatrix_square[idx] for idx in product(cluster01_members, cluster03_members)])
        assert np.all(cluster01_against_cluster03 <= 0.50), cluster01_against_cluster03

        cluster02_against_cluster03 = np.array([simmatrix_square[idx] for idx in product(cluster02_members, cluster03_members)])
        assert np.all(cluster02_against_cluster03 <= 0.50), cluster02_against_cluster03

    def test_real_data_ps_vo_as_ndarray(self):
        ## Prepare
        ps_data_file = get_data_file('ps_2011_2012.csv')
        vo_data_file = get_data_file('vo_2011_2012.csv')

        sources = merge_sources([ps_data_file, vo_data_file])[0]
        n_objects = 10 + 42

        ## Run
        simmatrix = calculate_simmatrix(sources.values, internal_n_clusters=(2,))

        ## Validate
        assert simmatrix is not None
        assert hasattr(simmatrix, 'shape')
        assert len(simmatrix.shape) == 1
        assert simmatrix.shape[0] == n_objects * (n_objects - 1) / 2

        assert not hasattr(simmatrix, 'columns')
        assert not hasattr(simmatrix, 'index')

    def test_real_data_ps_vo_as_dataframe(self):
        ## Prepare
        ps_data_file = get_data_file('ps_2011_2012.csv')
        vo_data_file = get_data_file('vo_2011_2012.csv')

        sources = merge_sources([ps_data_file, vo_data_file])[0]
        n_objects = 10 + 42

        ## Run
        simmatrix = calculate_simmatrix(sources, internal_n_clusters=(2,))

        ## Validate
        assert simmatrix is not None
        assert hasattr(simmatrix, 'shape')
        assert len(simmatrix.shape) == 2
        assert simmatrix.shape == (n_objects, n_objects)

        assert np.all(np.diag(simmatrix) == 1.0)
        assert np.all(np.triu(simmatrix) == np.tril(simmatrix).T)

        assert hasattr(simmatrix, 'columns')
        assert hasattr(simmatrix, 'index')

        assert simmatrix.index.dtype == 'object'

        assert simmatrix.loc['Arom-1'] is not None
        assert len(simmatrix.loc['Arom-1']) == n_objects

        assert simmatrix['PENTENONA'] is not None
        assert len(simmatrix['PENTENONA']) == n_objects

        assert simmatrix.loc['Dulz-5', 'UNK 57-3'] is not None
        assert simmatrix.loc['Dulz-5', 'UNK 57-3'] > 0.0
        assert simmatrix.loc['UNK 57-3', 'Dulz-5'] == simmatrix.loc['Dulz-5', 'UNK 57-3']

    def test_invalid_internal_clustering_method00(self):
        ## Prepare
        ps_data_file = get_data_file('ps_2011_2012.csv')
        vo_data_file = get_data_file('vo_2011_2012.csv')

        sources = merge_sources([ps_data_file, vo_data_file])[0]

        ## Run
        try:
            calculate_simmatrix(sources, internal_clustering_method='invalid_method')
        except ValueError:
            pass
        else:
            raise AssertionError('Failure expected')

    def test_clustermatch_a_pair_of_features_share_no_objects(self):
        # prepare
        processed_features = merge_sources(get_data_file('2008-2009_no_obj_shared.xls'))[0]

        # run
        sim_matrix = calculate_simmatrix(processed_features, internal_n_clusters=2)

        # validate
        assert sim_matrix is not None
        assert sim_matrix.loc['Carotenes', '1-fenil-1-propanol'] == 0

    def test_clustermatch_a_pair_of_features_share_only_one_object(self):
        # prepare
        processed_features = merge_sources(get_data_file('2008-2009_one_obj_shared.xls'))[0]

        # run
        sim_matrix = calculate_simmatrix(processed_features, internal_n_clusters=2)

        # validate
        assert sim_matrix is not None
        assert sim_matrix.loc['4-metil-3-hepten-2-ona', 'Citrate'] == 0

    def test_clustermatch_a_pair_of_features_share_only_two_object(self):
        # prepare
        processed_features = merge_sources(get_data_file('2008-2009_two_obj_shared.xls'))[0]

        # run
        sim_matrix = calculate_simmatrix(processed_features, internal_n_clusters=2)

        # validate
        assert sim_matrix is not None
        assert sim_matrix.loc['1-fenil-1-propanol', 'Ethanol'] == 0

    def test_artificial_data_complete_multiprocess_correctness(self):
        ## Prepare
        np.random.seed(0)

        # source01
        source01 = np.array([
            [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  #00 1 <<
            [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  #01 1 <<
            [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  #02 1 <<
            [0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],  #03 2 --
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #04 2 --
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  #05 3
        ], dtype=np.float)
        # source01 = source01 + np.random.rand(6, 7) * 0.2

        # source02
        source02 = np.array([
            [100, 100,   0,    0,   0,  0,    0.0],  #06 2
            [ 50,  50,   0,    0,   0,  0,    0.0],  #07 2 --
            [  0,   0,   0,    0,   0,  0,    0.1],  #08 3
            [900, 900,   0,    0,   0,  0,    0.0],  #09 2 --
            [  0,   0,  50,  100,  50,  0,    0.0],  #10 1 <<
            [  0,   0,   0,    0,   0,  0, 1000.0],  #11 3
            [  0,   0, 500, 1000, 500,  0,    0.0],  #12 1 <<
            [100, 100,   0,    0,   0,  0,    0.0],  #13 2 --
            [  0,   0,   0,    0,   0,  0,  100.0],  #14 3
        ], dtype=np.float)
        # source02 = source02 + np.random.rand(9, 7) * 20

        # source03
        source03 = np.array([
            [100, 100,   0,    0,   0,  0,    0.0],  #15 2
            [ 50,  50,   0,    0,   0,  0,    0.0],  #16 2 --
            [  0,   0,   0,    0,   0,  0,    0.1],  #17 3
            [900, 900,   0,    0,   0,  0,    0.0],  #18 2 --
            [  0,   0,  50,  100,  50,  0,    0.0],  #19 1 <<
            [  0,   0,   0,    0,   0,  0, 1000.0],  #20 3
            [  0,   0, 500, 1000, 500,  0,    0.0],  #21 1 <<
            [100, 100,   0,    0,   0,  0,    0.0],  #22 2 --
            [  0,   0,   0,    0,   0,  0,  100.0],  #23 3
        ], dtype=np.float)

        # source04
        source04 = np.array([
            [100, 100,   0,    0,   0,  0,    0.0],  #24 2
            [ 50,  50,   0,    0,   0,  0,    0.0],  #25 2 --
            [  0,   0,   0,    0,   0,  0,    0.1],  #26 3
            [900, 900,   0,    0,   0,  0,    0.0],  #27 2 --
            [  0,   0,  50,  100,  50,  0,    0.0],  #28 1 <<
            [  0,   0,   0,    0,   0,  0, 1000.0],  #29 3
            [  0,   0, 500, 1000, 500,  0,    0.0],  #30 1 <<
            [100, 100,   0,    0,   0,  0,    0.0],  #31 2 --
            [  0,   0,   0,    0,   0,  0,  100.0],  #32 3
        ], dtype=np.float)

        data = np.concatenate((source01, source02, source03, source04))

        ## Run
        simmatrix = calculate_simmatrix(data, internal_n_clusters=2, n_jobs=-1)

        ## Validate
        assert simmatrix is not None
        assert hasattr(simmatrix, 'shape')
        assert len(simmatrix.shape) == 1
        assert simmatrix.shape[0] == 528

        simmatrix_square = get_squareform(simmatrix)
        assert simmatrix_square.shape == (33, 33)
        assert np.all(np.diag(simmatrix_square) == 1.0)

        cluster01_members = (0, 1, 2, 10, 12, 19, 21, 28, 30)
        cluster02_members = (3, 4, 6, 7, 9, 13, 15, 16, 18, 22, 24, 25, 27, 31)
        cluster03_members = (5, 8, 11, 14, 17, 20, 23, 26, 29, 32)

        assert len(np.unique(cluster01_members + cluster02_members + cluster03_members)) == 33

        cluster01_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster01_members, 2)])
        assert np.all(cluster01_sim == 1.0), cluster01_sim

        cluster02_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster02_members, 2)])
        assert np.all(cluster02_sim == 1.0), cluster02_sim

        cluster03_sim = np.array([simmatrix_square[idx] for idx in combinations(cluster03_members, 2)])
        assert np.all(cluster03_sim == 1.0), cluster03_sim

        cluster01_against_cluster02 = np.array([simmatrix_square[idx] for idx in product(cluster01_members, cluster02_members)])
        assert np.all(cluster01_against_cluster02 <= 0.50), cluster01_against_cluster02

        cluster01_against_cluster03 = np.array([simmatrix_square[idx] for idx in product(cluster01_members, cluster03_members)])
        assert np.all(cluster01_against_cluster03 <= 0.50), cluster01_against_cluster03

        cluster02_against_cluster03 = np.array([simmatrix_square[idx] for idx in product(cluster02_members, cluster03_members)])
        assert np.all(cluster02_against_cluster03 <= 0.50), cluster02_against_cluster03

    def test_artificial_data_complete_multiprocess_performance(self):
        # FIXME this test should be ignored if cpu_count < 2 or 4

        ## Prepare
        np.random.seed(0)
        data = np.random.rand(100, 20)

        ## Run
        # first, with no paralellization
        start_time = time()
        calculate_simmatrix(data, internal_n_clusters=5, n_jobs=1)
        no_parallel_time = time() - start_time

        start_time = time()
        calculate_simmatrix(data, internal_n_clusters=5, n_jobs=4)
        parallel_time = time() - start_time

        ## Validate
        # parallel_time should be half of no_parallel_time, but sometimes it's higher
        assert parallel_time * 2.0 <= no_parallel_time, (parallel_time, no_parallel_time)

    def test_numerical_and_categorical_data01(self):
        ## Prepare
        num_data = np.random.rand(10, 20)

        last_part = run_quantile_clustering(num_data[0], 2)

        cat_data = np.array(['cat0{}'.format(c) for c in last_part])

        data = np.array([
            *num_data,
            cat_data,
        ],
        dtype=object)
        n_objects = 11

        ## Run
        simmatrix = calculate_simmatrix(data, n_jobs=1)

        ## Validate
        assert simmatrix is not None
        assert hasattr(simmatrix, 'shape')
        assert len(simmatrix.shape) == 1
        assert simmatrix.shape[0] == 55

        simmatrix_square = get_squareform(simmatrix)
        assert simmatrix_square.shape == (n_objects, n_objects)
        assert np.all(np.diag(simmatrix_square) == 1.0)

        assert simmatrix_square[0, 10] == 1.0
        assert simmatrix_square[0, 9] < 0.1

    def test_repeated_kmeans_run_same_parameters_check_sim_matrix(self):
        # prepare
        processed_features = merge_sources(get_data_file('sample_tiny.xls'))[0]

        # run
        # first round
        np.random.seed(0)

        sim_matrix_0_0 = \
            calculate_simmatrix(processed_features, internal_n_clusters=range(2, 4),
                                internal_clustering_method='kmeans',
                                kmeans_n_init=3, kmeans_random_state=0)

        sim_matrix_1_0 = \
            calculate_simmatrix(processed_features, internal_n_clusters=range(2, 4),
                                internal_clustering_method='kmeans', n_jobs=4,
                                kmeans_n_init=10, kmeans_random_state=0)

        # second round
        np.random.seed(1)

        sim_matrix_0_1 = \
            calculate_simmatrix(processed_features, internal_n_clusters=range(2, 4),
                                internal_clustering_method='kmeans', n_jobs=4,
                                kmeans_n_init=3, kmeans_random_state=1)

        sim_matrix_1_1 = \
            calculate_simmatrix(processed_features, internal_n_clusters=range(2, 4),
                                internal_clustering_method='kmeans', n_jobs=4,
                                kmeans_n_init=10, kmeans_random_state=1)

        sim_matrix_n_cells = processed_features.shape[0] ** 2

        # validate
        # check indexes
        sm1idx = sorted(sim_matrix_0_0.index)
        sm2idx = sorted(sim_matrix_0_1.index)
        assert sm1idx == sm2idx

        # check columns
        sm1cols = sorted(sim_matrix_0_0.columns)
        sm2cols = sorted(sim_matrix_0_1.columns)
        assert sm1cols == sm2cols

        # for 0
        notequal_cells = (sim_matrix_0_0 != sim_matrix_0_1)
        notequal_cells_count = notequal_cells.sum().sum()
        notequal_cells_ratio_0 = notequal_cells_count / sim_matrix_n_cells
        assert notequal_cells_ratio_0 < 0.55, notequal_cells_ratio_0
        diffs = (sim_matrix_0_0 - sim_matrix_0_1).abs()
        max_diff = diffs.max().max()
        assert max_diff < 0.47, max_diff

        # for 1
        notequal_cells = (sim_matrix_1_0 != sim_matrix_1_1)
        notequal_cells_count = notequal_cells.sum().sum()
        notequal_cells_ratio_1 = notequal_cells_count / sim_matrix_n_cells
        # assert k5_notequal_cells_ratio < 0.30, k5_notequal_cells_ratio
        assert notequal_cells_ratio_1 < 0.25 < notequal_cells_ratio_0, notequal_cells_ratio_1
        diffs = (sim_matrix_1_0 - sim_matrix_1_1).abs()
        max_diff = diffs.max().max()
        assert max_diff < 0.14, max_diff

    def test_quantiles_same_rows_but_in_different_sources(self):
        # prepare
        data0 = merge_sources(get_data_file('sampleAA.xlsx'))[0]
        data1 = merge_sources(get_data_file('sampleBB.xlsx'))[0]

        # run
        np.random.seed(0)

        sm0 = \
            calculate_simmatrix(data0, internal_n_clusters=range(2,4),
                                internal_clustering_method='quantiles')

        sm1 = \
            calculate_simmatrix(data1, internal_n_clusters=range(2,4),
                                internal_clustering_method='quantiles')

        # validate
        assert len(sm0.index) == len(sm1.index), 'len indexes does not match'
        assert len(sm0.columns) == len(sm1.columns), 'len columns does not match'

        # plain indexes and columns should not match
        assert sm0.index.tolist() != sm1.index.tolist()
        assert sm0.columns.tolist() != sm1.columns.tolist()

        sm0_s_index = sorted(sm0.index)
        sm0_s_columns = sorted(sm0.columns)

        sm1_s_index = sorted(sm1.index)
        sm1_s_columns = sorted(sm1.columns)

        # sorted indexes and columns should match
        assert sm0_s_index == sm1_s_index, 'indexes does not match'
        assert sm0_s_columns == sm1_s_columns, 'columns does not match'

        diffs = (sm0.loc[sm0_s_index, sm0_s_columns] != sm1.loc[sm1_s_index, sm1_s_columns]) & \
                ~np.isnan(sm0.loc[sm0_s_index, sm0_s_columns]) & \
                ~np.isnan(sm1.loc[sm1_s_index, sm1_s_columns])

        assert not diffs.any().any()

    def test_quantiles_same_data_shuffled_objects(self):
        # prepare
        data0 = merge_sources([get_data_file('sampleAA.xlsx')])[0]

        # run
        np.random.seed(0)
        data1 = data0.sample(frac=1)

        sm0 = \
            calculate_simmatrix(data0, internal_n_clusters=range(2, 20),
                                internal_clustering_method='quantiles', n_jobs=4)

        sm1 = \
            calculate_simmatrix(data1, internal_n_clusters=range(2, 20),
                                internal_clustering_method='quantiles', n_jobs=4)

        # validate
        assert len(sm0.index) == len(sm1.index), 'len indexes does not match'
        assert len(sm0.columns) == len(sm1.columns), 'len columns does not match'

        # plain indexes and columns should not match
        assert sm0.index.tolist() != sm1.index.tolist()
        assert sm0.columns.tolist() != sm1.columns.tolist()

        sm0_s_index = sorted(sm0.index)
        sm0_s_columns = sorted(sm0.columns)

        sm1_s_index = sorted(sm1.index)
        sm1_s_columns = sorted(sm1.columns)

        # sorted indexes and columns should match
        assert sm0_s_index == sm1_s_index, 'indexes does not match'
        assert sm0_s_columns == sm1_s_columns, 'columns does not match'

        assert sm0.loc[sm0_s_index, sm0_s_columns].equals(sm1.loc[sm1_s_index, sm1_s_columns])
