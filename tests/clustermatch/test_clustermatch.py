import unittest

import numpy as np
from sklearn.metrics import adjusted_rand_score as ari

from clustermatch.cluster import clustermatch, calculate_simmatrix, get_partition_agglomerative, get_partition_spectral
from clustermatch.utils.data import merge_sources
from tests.utils import get_data_file


class ClusterMatchTest(unittest.TestCase):
    def test_particion_shape(self):
        ## Prepare
        np.random.seed(33)

        # la fuente01 tiene 10 mediciones y 50 tomates
        fuente01 = np.random.rand(10, 50)
        fuente02 = np.random.rand(7, 50)

        ## Run
        salida = clustermatch(np.concatenate((fuente01, fuente02)), 3)[0]

        ## Validar
        assert salida is not None
        assert hasattr(salida, 'shape')
        assert len(salida.shape) == 1
        assert len(salida) == 10 + 7

    def test_datos_artificiales_diego_completos(self):
        """
        Este test usa los datos artificiales de Diego, los originales
        sin datos faltantes.
        :return:
        """
        ## Preparar
        np.random.seed(33)

        # antioxidantes
        ao = \
            [[0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  # 1 <<
             [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  # 1 <<
             [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  # 1 <<
             [0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 --
             [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 --
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]  # 3
        ao = ao + np.random.rand(6, 7) * 0.2

        # metabolitos
        me = [[100, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ~2
              [50, 50, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 --
              [50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ~2
              [50, 50, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 --
              [0.0, 0.0, 50, 100, 50, 0.0, 0.0],  # 1 <<
              [0.0, 50, 0.0, 0.0, 0.0, 0.0, 100],  # ~3~2
              [0.0, 0.0, 50, 100, 50, 0.0, 0.0],  # 1 <<
              [100, 100, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 --
              [0.0, 100, 0.0, 0.0, 0.0, 0.0, 100]]  # ~3~2
        me = me + np.random.rand(9, 7) * 20

        particion_referencia = np.array([2, 2, 2, 1, 1, 0, 1, 1, 1, 1, 2, 0, 2, 1, 0])

        ## Correr
        salida = clustermatch(np.concatenate((ao, me)), 3, internal_n_clusters=range(2, 3), internal_clustering_method='kmeans')[0]

        ## Validar
        assert len(np.unique(salida)) == 3

        ari_val = ari(particion_referencia, salida)
        assert ari_val == 1.0, ari_val

    def test_datos_artificiales_diego_algunos_nan_en_columna(self):
        """
        En este test se usan los mismos datos originales generados por Diego,
        pero para un "tomate" de la segunda fuente tiene algunos datos faltantes.
        :return:
        """
        ## Preparar
        np.random.seed(33)

        # antioxidantes
        ao = \
            [[0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  # 1 <<
             [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  # 1 <<
             [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  # 1 <<
             [0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 --
             [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 --
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]  # 3
        ao = ao + np.random.rand(6, 7) * 0.2

        # metabolitos
        # solo algunos valores de la misma columna son np.nan
        me = [[100.0,   0.0,  0.0,   0.0,  0.0, np.nan,   0.0],  # ~2
              [ 50.0,  50.0,  0.0,   0.0,  0.0,    0.0,   0.0],  # 2 --
              [ 50.0,   0.0,  0.0,   0.0,  0.0, np.nan,   0.0],  # ~2
              [ 50.0,  50.0,  0.0,   0.0,  0.0, np.nan,   0.0],  # 2 --
              [  0.0,   0.0, 50.0, 100.0, 50.0,    0.0,   0.0],  # 1 <<
              [  0.0,  50.0,  0.0,   0.0,  0.0, np.nan, 100.0],  # ~3~2
              [  0.0,   0.0, 50.0, 100.0, 50.0, np.nan,   0.0],  # 1 <<
              [100.0, 100.0,  0.0,   0.0,  0.0,    0.0,   0.0],  # 2 --
              [  0.0, 100.0,  0.0,   0.0,  0.0, np.nan, 100.0]]  # ~3~2
        me = me + np.random.rand(9, 7) * 20

        particion_referencia = np.array([2, 2, 2, 1, 1, 0, 1, 1, 1, 1, 2, 0, 2, 1, 0])

        ## Correr
        salida = clustermatch(np.concatenate((ao, me)), 3, internal_n_clusters=range(2, 3), internal_clustering_method='kmeans')[0]

        ## Validar
        ari_val = ari(particion_referencia, salida)
        assert ari_val == 1.0, ari_val

    def test_datos_artificiales_diego_columna_faltante(self):
        """
        En este test se usan los mismos datos originales generados por Diego,
        pero para la segunda fuente no hay datos de un tomate.
        :return:
        """
        ## Preparar
        np.random.seed(33)

        # antioxidantes
        ao = \
            [[0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  # 1 <<
             [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  # 1 <<
             [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  # 1 <<
             [0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 --
             [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 --
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]  # 3
        ao = ao + np.random.rand(6, 7) * 0.2

        # metabolitos
        # esta característica no tiene el tomate 6
        me = [[100.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0],  # ~2
              [50.0, 50.0, 0.0, 0.0, 0.0, np.nan, 0.0],  # 2 --
              [50.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0],  # ~2
              [50.0, 50.0, 0.0, 0.0, 0.0, np.nan, 0.0],  # 2 --
              [0.0, 0.0, 50.0, 100.0, 50.0, np.nan, 0.0],  # 1 <<
              [0.0, 50.0, 0.0, 0.0, 0.0, np.nan, 100.0],  # ~3~2
              [0.0, 0.0, 50.0, 100.0, 50.0, np.nan, 0.0],  # 1 <<
              [100.0, 100.0, 0.0, 0.0, 0.0, np.nan, 0.0],  # 2 --
              [0.0, 100.0, 0.0, 0.0, 0.0, np.nan, 100.0]]  # ~3~2
        me = me + np.random.rand(9, 7) * 20

        particion_referencia = np.array([2, 2, 2, 1, 1, 0, 1, 1, 1, 1, 2, 0, 2, 1, 0])

        ## Correr
        salida = clustermatch(np.concatenate((ao, me)), 3, internal_n_clusters=range(2, 3), internal_clustering_method='kmeans')[0]

        ## Validar
        ari_val = ari(particion_referencia, salida)
        assert ari_val == 1.0, ari_val

    def test_datos_artificiales_diego_columna_faltante_y_otros_nan(self):
        """
        En este test hay más datos faltantes. Además de que la segunda fuente
        no tiene los datos de un tomate en particular, ésta tampoco tiene algunos
        datos de otros dos tomates.
        :return:
        """
        ## Preparar
        np.random.seed(33)

        # antioxidantes
        ao = \
            [[0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  # 1 <<
             [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  # 1 <<
             [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  # 1 <<
             [0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 --
             [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 --
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]  # 3
        ao = ao + np.random.rand(6, 7) * 0.2

        # metabolitos
        # esta característica no tiene el tomate 6
        me = [[100.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0],  # ~2
              [50.0, 50.0, 0.0, 0.0, 0.0, np.nan, 0.0],  # 2 --
              [50.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0],  # ~2
              [50.0, 50.0, 0.0, 0.0, 0.0, np.nan, 0.0],  # 2 --
              [0.0, 0.0, np.nan, np.nan, 50.0, np.nan, 0.0],  # 1 <<
              [0.0, 50.0, 0.0, 0.0, 0.0, np.nan, 100.0],  # ~3~2
              [0.0, 0.0, 50.0, 100.0, 50.0, np.nan, 0.0],  # 1 <<
              [100.0, 100.0, 0.0, 0.0, 0.0, np.nan, 0.0],  # 2 --
              [0.0, 100.0, 0.0, 0.0, 0.0, np.nan, 100.0]]  # ~3~2
        me = me + np.random.rand(9, 7) * 20

        particion_referencia = np.array([2, 2, 2, 1, 1, 0, 1, 1, 1, 1, 2, 0, 2, 1, 0])

        ## Correr
        salida = clustermatch(np.concatenate((ao, me)), 3, internal_n_clusters=range(2, 3), internal_clustering_method='kmeans')[0]

        ## Validar
        ari_val = ari(particion_referencia, salida)
        assert ari_val == 1.0, ari_val

    def test_datos_artificiales_diego_dos_fuentes_con_nan(self):
        """
        En este test hay más datos faltantes, ahora también en la primer fuente.
        :return:
        """
        ## Preparar
        np.random.seed(33)

        # antioxidantes
        ao = \
            [[0.0, 0.0, 0.5, 1.0, np.nan, 0.0, 0.0],  # 1 <<
             [0.0, 0.0, 0.5, np.nan, 0.5, 0.0, 0.0],  # 1 <<
             [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  # 1 <<
             [0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 --
             [1.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 --
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]  # 3
        ao = ao + np.random.rand(6, 7) * 0.2

        # metabolitos
        # esta característica no tiene el tomate 6
        me = [[100.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0],  # ~2
              [50.0, 50.0, 0.0, 0.0, 0.0, np.nan, 0.0],  # 2 --
              [50.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0],  # ~2
              [50.0, 50.0, 0.0, 0.0, 0.0, np.nan, 0.0],  # 2 --
              [0.0, 0.0, np.nan, np.nan, 50.0, np.nan, 0.0],  # 1 <<
              [0.0, 50.0, 0.0, 0.0, 0.0, np.nan, 100.0],  # ~3~2
              [0.0, 0.0, 50.0, 100.0, 50.0, np.nan, 0.0],  # 1 <<
              [100.0, 100.0, 0.0, 0.0, 0.0, np.nan, 0.0],  # 2 --
              [0.0, 100.0, 0.0, 0.0, 0.0, np.nan, 100.0]]  # ~3~2
        me = me + np.random.rand(9, 7) * 20

        particion_referencia = np.array([2, 2, 2, 1, 1, 0, 1, 1, 1, 1, 2, 0, 2, 1, 0])

        ## Correr
        salida = clustermatch(np.concatenate((ao, me)), 3, internal_clustering_method='kmeans')[0]

        ## Validar
        assert ari(particion_referencia, salida) == 1.0

    def test_clustermatch_ps_vo(self):
        ## Prepare
        np.random.seed(33)

        ps_data_file = get_data_file('ps_2011_2012.csv')
        vo_data_file = get_data_file('vo_2011_2012.csv')

        sources = merge_sources([ps_data_file, vo_data_file])[0]

        ## Run
        partition = clustermatch(sources.values, 10, internal_n_clusters=(2,3))[0]

        ## Validate
        # FIXME find a way to validate results with these data
        assert partition is not None
        assert len(partition) == 10 + 42  # + 89  # cantidad de objetos
        assert len(np.unique(partition)) == 10  # número de clusters

    def test_clustermatch_using_spectral_clustering(self):
        ## Prepare
        ps_data_file = get_data_file('ps_2011_2012.csv')
        vo_data_file = get_data_file('vo_2011_2012.csv')

        sources = merge_sources([ps_data_file, vo_data_file])[0]

        ## Run
        np.random.seed(0)
        sim_matrix = calculate_simmatrix(sources.values, internal_n_clusters=(2,3))
        partition = get_partition_spectral(sim_matrix, n_clusters=10)

        ## Validate
        assert partition is not None
        assert len(partition) == 10 + 42  # + 89  # n objects
        assert len(np.unique(partition)) == 10  # n clusters

    def test_clustermatch_using_agglomerative_returns_linkage_matrix(self):
        ## Prepare
        ps_data_file = get_data_file('ps_2011_2012.csv')
        vo_data_file = get_data_file('vo_2011_2012.csv')

        sources = merge_sources([ps_data_file, vo_data_file])[0]
        n_objects = sources.shape[0]

        # get partition with spectral to compare with
        sim_matrix = calculate_simmatrix(sources.values, internal_n_clusters=(2,3))

        ## Run
        partition, linkage_matrix = get_partition_agglomerative(sim_matrix, n_clusters=10, linkage_method='average')

        ## Validate
        assert partition is not None
        assert len(partition) == 10 + 42  # + 89  # n objects
        assert len(np.unique(partition)) == 10  # n clusters

        assert linkage_matrix is not None
        assert hasattr(linkage_matrix, 'shape')
        assert linkage_matrix.shape == (n_objects - 1, 4)

    def test_clustermatch_using_agglomerative_clustering_average(self):
        ## Prepare
        ps_data_file = get_data_file('ps_2011_2012.csv')
        vo_data_file = get_data_file('vo_2011_2012.csv')

        sources = merge_sources([ps_data_file, vo_data_file])[0]

        # get partition with spectral to compare with
        np.random.seed(0)
        sim_matrix = calculate_simmatrix(sources.values, internal_n_clusters=(2,3))
        partition_spectral = get_partition_spectral(sim_matrix, n_clusters=10)

        ## Run
        np.random.seed(0)
        partition = get_partition_agglomerative(sim_matrix, n_clusters=10, linkage_method='average')[0]

        ## Validate
        assert partition is not None
        assert len(partition) == 10 + 42  # + 89  # n objects
        assert len(np.unique(partition)) == 10  # n clusters
        assert ari(partition_spectral, partition) < 1.0  # spectral and agglomerative partitions should be different

    def test_clustermatch_using_agglomerative_clustering_single(self):
        ## Prepare
        ps_data_file = get_data_file('ps_2011_2012.csv')
        vo_data_file = get_data_file('vo_2011_2012.csv')

        sources = merge_sources([ps_data_file, vo_data_file])[0]

        # get partition with spectral to compare with
        np.random.seed(0)
        sim_matrix = calculate_simmatrix(sources.values, internal_n_clusters=range(2, 3))
        reference_partition = get_partition_agglomerative(sim_matrix, n_clusters=5, linkage_method='average')[0]

        ## Run
        np.random.seed(0)
        partition = get_partition_agglomerative(sim_matrix, n_clusters=5, linkage_method='single')[0]

        ## Validate
        assert partition is not None
        assert len(partition) == 10 + 42  # + 89  # n objects
        assert len(np.unique(partition)) == 5  # n clusters
        assert ari(reference_partition, partition) < 1.0  # reference and agglomerative partitions should be different

    def test_clustermatch_using_agglomerative_clustering_complete(self):
        ## Prepare
        ps_data_file = get_data_file('ps_2011_2012.csv')
        vo_data_file = get_data_file('vo_2011_2012.csv')

        sources = merge_sources([ps_data_file, vo_data_file])[0]

        # get partition with spectral to compare with
        np.random.seed(0)
        sim_matrix = calculate_simmatrix(sources.values, internal_n_clusters=range(2, 3))
        reference_partition = get_partition_agglomerative(sim_matrix, n_clusters=5, linkage_method='single')[0]

        ## Run
        np.random.seed(0)
        partition = get_partition_agglomerative(sim_matrix, n_clusters=5, linkage_method='complete')[0]

        ## Validate
        assert partition is not None
        assert len(partition) == 10 + 42  # + 89  # n objects
        assert len(np.unique(partition)) == 5  # n clusters
        assert ari(reference_partition, partition) < 1.0  # reference and agglomerative partitions should be different

    def test_clustermatch_using_agglomerative_clustering_ward(self):
        ## Prepare
        ps_data_file = get_data_file('ps_2011_2012.csv')
        vo_data_file = get_data_file('vo_2011_2012.csv')

        sources = merge_sources([ps_data_file, vo_data_file])[0]

        # get partition with spectral to compare with
        np.random.seed(0)
        sim_matrix = calculate_simmatrix(sources.values, internal_n_clusters=range(2, 3))
        reference_partition = get_partition_agglomerative(sim_matrix, n_clusters=10, linkage_method='complete')[0]

        ## Run
        np.random.seed(0)
        partition = get_partition_agglomerative(sim_matrix, n_clusters=10, linkage_method='ward')[0]

        ## Validate
        assert partition is not None
        assert len(partition) == 10 + 42  # + 89  # n objects
        assert len(np.unique(partition)) == 10  # n clusters
        assert ari(reference_partition, partition) < 1.0  # reference and agglomerative partitions should be different

    def test_clustermatch_multiple_k_values(self):
        ## Prepare
        ps_data_file = get_data_file('ps_2011_2012.csv')
        vo_data_file = get_data_file('vo_2011_2012.csv')

        sources = merge_sources([ps_data_file, vo_data_file])[0]

        ## Run
        sim_matrix = calculate_simmatrix(sources.values, internal_n_clusters=range(2, 3))
        partition = get_partition_agglomerative(sim_matrix, n_clusters=(9, 10, 11), linkage_method='average')[0]

        ## Validate
        assert partition is not None
        assert hasattr(partition, 'shape')
        assert partition.shape == (10 + 42, 3)
        assert hasattr(partition, 'columns')
        assert 'k=9' in partition.columns
        assert 'k=10' in partition.columns
        assert 'k=11' in partition.columns

        assert len(partition['k=9'].unique()) == 9
        assert len(partition['k=10'].unique()) == 10
        assert len(partition['k=11'].unique()) == 11

        assert hasattr(partition, 'index')
        assert partition.index.dtype == 'int64'

    def test_clustermatch_multiple_k_values_measures_names_are_kept_in_partition(self):
        ## Prepare
        ps_data_file = get_data_file('ps_2011_2012.csv')
        vo_data_file = get_data_file('vo_2011_2012.csv')

        sources = merge_sources([ps_data_file, vo_data_file])[0]

        ## Run
        sim_matrix = calculate_simmatrix(sources, internal_n_clusters=range(2, 3))
        partition = get_partition_spectral(sim_matrix, n_clusters=(9, 10, 11))

        ## Validate
        assert partition is not None
        assert hasattr(partition, 'shape')
        assert partition.shape == (10 + 42, 3)
        assert hasattr(partition, 'columns')
        assert 'k=9' in partition.columns
        assert 'k=10' in partition.columns
        assert 'k=11' in partition.columns

        assert len(partition['k=9'].unique()) == 9
        assert len(partition['k=10'].unique()) == 10
        assert len(partition['k=11'].unique()) == 11

        # measure labels were kept
        assert hasattr(partition, 'index')
        assert partition.index.dtype == 'object'

        assert partition.loc['Arom-1'] is not None
        assert len(partition.loc['Arom-1']) == 3

    def test_clustermatch_single_k_value_measures_names_are_kept_in_partition(self):
        ## Prepare
        ps_data_file = get_data_file('ps_2011_2012.csv')
        vo_data_file = get_data_file('vo_2011_2012.csv')

        sources = merge_sources([ps_data_file, vo_data_file])[0]

        ## Run
        sim_matrix = calculate_simmatrix(sources, internal_n_clusters=range(2, 3))
        partition = get_partition_spectral(sim_matrix, n_clusters=9)

        ## Validate
        assert partition is not None
        assert hasattr(partition, 'shape')
        assert partition.shape == (10 + 42, 1)
        assert 'k=9' in partition.columns

        assert len(partition['k=9'].unique()) == 9

        # measure labels were kept
        assert hasattr(partition, 'index')
        assert partition.index.dtype == 'object'

        assert partition.loc['Arom-1'] is not None
        assert len(partition.loc['Arom-1']) == 1
