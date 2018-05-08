import unittest
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.special._ufuncs import boxcox

from experiments.data import blobs_data_generator01, transform_rows_full_scaled01, transform_rows_nonlinear01, \
    transform_rows_nonlinear02, transform_rows_nonlinear03, transform_rows_boxcox01, transform_rows_boxcox02, \
    transform_rows_boxcox03, transform_rows_boxcox04, transform_rows_boxcox05, transform_rows_boxcox06, \
    transform_rows_boxcox07, transform_rows_boxcox_max_distance2, transform_rows_boxcox_max_distance12


class ExperimentDataTest(unittest.TestCase):
    def test_blob_data_generator(self):
        # Run
        data, data_ref = blobs_data_generator01()

        # Validate
        assert data is not None
        assert hasattr(data, 'shape')
        assert data.shape == (100, 20)

        # data has values around -1.0 and 1.0
        assert np.all(data >= -1.30) and np.all(data <= 1.30)

        assert data_ref is not None
        assert hasattr(data_ref, 'shape')
        assert data_ref.shape == (100,)
        assert len(np.unique(data_ref)) == 3

        # check that clusters are compact and sufficiently well separated
        cluster0 = data[data_ref == 0]
        cluster0_inner_max = np.max([euclidean(point0, point1) for point0 in cluster0 for point1 in cluster0])

        cluster1 = data[data_ref == 1]
        cluster1_inner_max = np.max([euclidean(point0, point1) for point0 in cluster1 for point1 in cluster1])

        cluster2 = data[data_ref == 2]
        cluster2_inner_max = np.max([euclidean(point0, point1) for point0 in cluster2 for point1 in cluster2])

        cluster0_cluster1_min_distance = np.min([euclidean(point_c0, point_c1) for point_c0 in cluster0 for point_c1 in cluster1])
        cluster0_cluster2_min_distance = np.min([euclidean(point_c0, point_c2) for point_c0 in cluster0 for point_c2 in cluster2])
        cluster1_cluster2_min_distance = np.min([euclidean(point_c1, point_c2) for point_c1 in cluster1 for point_c2 in cluster2])

        # maximum distance within cluster points should be half the min distance between clusters points
        assert cluster0_inner_max * 2 < cluster0_cluster1_min_distance
        assert cluster0_inner_max * 2 < cluster0_cluster2_min_distance

        assert cluster1_inner_max * 2 < cluster0_cluster1_min_distance
        assert cluster1_inner_max * 2 < cluster1_cluster2_min_distance

        assert cluster2_inner_max * 2 < cluster1_cluster2_min_distance
        assert cluster2_inner_max * 2 < cluster0_cluster2_min_distance

        # check that clusters are shuffled
        cluster0_idxs = np.where(data_ref == 0)[0]
        cluster1_idxs = np.where(data_ref == 1)[0]
        cluster2_idxs = np.where(data_ref == 2)[0]

        assert np.any(cluster0_idxs > cluster1_idxs.min()) and np.any(cluster1_idxs > cluster0_idxs.min()), 'clusters are not shuffled'
        assert np.any(cluster0_idxs > cluster2_idxs.min()) and np.any(cluster2_idxs > cluster0_idxs.min()), 'clusters are not shuffled'
        assert np.any(cluster1_idxs > cluster2_idxs.min()) and np.any(cluster2_idxs > cluster1_idxs.min()), 'clusters are not shuffled'

    def test_transform_rows_full_scaled01(self):
        # Prepare
        data, data_ref = blobs_data_generator01()

        # Run
        t_data = transform_rows_full_scaled01(data)

        # Validate
        assert t_data is not None
        assert hasattr(t_data, 'shape')
        assert t_data.shape == data.shape

        assert t_data[0:20].min() == data[0:20].min() * 0.01
        assert t_data[0:20].max() == data[0:20].max() * 0.01

        assert t_data[20:40].min() == data[20:40].min() * 0.1
        assert t_data[20:40].max() == data[20:40].max() * 0.1

        assert t_data[40:60].min() == data[40:60].min() * 10
        assert t_data[40:60].max() == data[40:60].max() * 10

        assert t_data[60:80].min() == data[60:80].min() * 100
        assert t_data[60:80].max() == data[60:80].max() * 100

        assert t_data[80:100].min() == data[80:100].min() * 1000
        assert t_data[80:100].max() == data[80:100].max() * 1000

    def test_transform_rows_nonlinear01(self):
        # Prepare
        data, data_ref = blobs_data_generator01()

        # Run
        t_data = transform_rows_nonlinear01(data)

        # Validate
        assert t_data is not None
        assert hasattr(t_data, 'shape')
        assert t_data.shape == data.shape

        assert not np.any(np.isnan(t_data))
        assert not np.any(np.isinf(t_data))

        assert np.all(t_data[0:20] == np.exp(data[0:20]))
        assert np.all(t_data[20:40] == np.power(data[20:40], 2))
        assert np.all(t_data[40:60] == np.log(np.abs(data[40:60])))
        assert np.all(t_data[60:80] == np.expm1(data[60:80]))
        assert np.all(t_data[80:100] == np.log10(np.abs(data[80:100])))

    def test_transform_rows_nonlinear02(self):
        # Prepare
        data, data_ref = blobs_data_generator01()

        # Run
        t_data = transform_rows_nonlinear02(data)

        # Validate
        assert t_data is not None
        assert hasattr(t_data, 'shape')
        assert t_data.shape == data.shape

        assert not np.any(np.isnan(t_data))
        assert not np.any(np.isinf(t_data))

        assert np.all(t_data[0:25] == np.power(data[0:25], 3))
        assert np.all(t_data[25:50] == np.log(np.abs(data[25:50])))
        assert np.all(t_data[50:75] == np.log1p(np.abs(data[50:75])))
        assert np.all(t_data[75:100] == np.exp2(data[75:100]))

    def test_transform_rows_nonlinear03(self):
        # Prepare
        data, data_ref = blobs_data_generator01()

        # Run
        t_data = transform_rows_nonlinear03(data)

        # Validate
        assert t_data is not None
        assert hasattr(t_data, 'shape')
        assert t_data.shape == data.shape

        assert not np.any(np.isnan(t_data))
        assert not np.any(np.isinf(t_data))

        assert np.all(t_data[0:10] == np.power(data[0:10], 4))
        assert np.all(t_data[10:20] == np.log(np.abs(data[10:20])))
        assert np.all(t_data[20:30] == np.exp2(data[20:30]))
        assert np.all(t_data[30:40] == data[30:40] * 100.0)
        assert np.all(t_data[40:50] == np.log1p(np.abs(data[40:50])))
        assert np.all(t_data[50:60] == np.power(data[50:60], 5))
        assert np.all(t_data[60:70] == data[60:70] * 10000.0)
        assert np.all(t_data[70:80] == np.log10(np.abs(data[70:80])))
        assert np.all(t_data[80:90] == data[80:90] * 0.0001)
        assert np.all(t_data[90:100] == np.log2(np.abs(data[90:100])))

    def test_transform_rows_boxcox01(self):
        # Prepare
        data, data_ref = blobs_data_generator01()

        # Run
        t_data = transform_rows_boxcox01(data)

        # Validate
        assert t_data is not None
        assert hasattr(t_data, 'shape')
        assert t_data.shape == data.shape

        assert not np.any(np.isnan(t_data))
        assert not np.any(np.isinf(t_data))

        # shift data to make it all positive
        pos_data = data + (-1 * data.min()) + 0.01

        assert np.all(t_data[0:20] == boxcox(pos_data[0:20], 0.00))
        assert np.all(t_data[20:40] == boxcox(pos_data[20:40], 0.25))
        assert np.all(t_data[40:60] == boxcox(pos_data[40:60], 0.50))
        assert np.all(t_data[60:80] == boxcox(pos_data[60:80], 0.75))
        assert np.all(t_data[80:100] == boxcox(pos_data[80:100], 1.00))

    def test_transform_rows_boxcox02(self):
        # Prepare
        data, data_ref = blobs_data_generator01()

        # Run
        t_data = transform_rows_boxcox02(data)

        # Validate
        assert t_data is not None
        assert hasattr(t_data, 'shape')
        assert t_data.shape == data.shape

        assert not np.any(np.isnan(t_data))
        assert not np.any(np.isinf(t_data))

        # shift data to make it all positive
        pos_data = data + (-1 * data.min()) + 0.01

        assert np.all(t_data[0:20] == boxcox(pos_data[0:20], 1.00))
        assert np.all(t_data[20:40] == boxcox(pos_data[20:40], 1.25))
        assert np.all(t_data[40:60] == boxcox(pos_data[40:60], 1.50))
        assert np.all(t_data[60:80] == boxcox(pos_data[60:80], 1.75))
        assert np.all(t_data[80:100] == boxcox(pos_data[80:100], 2.00))

    def test_transform_rows_boxcox03(self):
        # Prepare
        data, data_ref = blobs_data_generator01()

        # Run
        t_data = transform_rows_boxcox03(data)

        # Validate
        assert t_data is not None
        assert hasattr(t_data, 'shape')
        assert t_data.shape == data.shape

        assert not np.any(np.isnan(t_data))
        assert not np.any(np.isinf(t_data))

        # shift data to make it all positive
        pos_data = data + (-1 * data.min()) + 0.01

        assert np.all(t_data[0:10] == boxcox(pos_data[0:10], 2.00))
        assert np.all(t_data[10:20] == boxcox(pos_data[10:20], 3.00))
        assert np.all(t_data[20:30] == boxcox(pos_data[20:30], 4.00))
        assert np.all(t_data[30:40] == boxcox(pos_data[30:40], 5.00))
        assert np.all(t_data[40:50] == boxcox(pos_data[40:50], 6.00))
        assert np.all(t_data[50:60] == boxcox(pos_data[50:60], 7.00))
        assert np.all(t_data[60:70] == boxcox(pos_data[60:70], 8.00))
        assert np.all(t_data[70:80] == boxcox(pos_data[70:80], 9.00))
        assert np.all(t_data[80:90] == boxcox(pos_data[80:90], 10.00))
        assert np.all(t_data[90:100] == boxcox(pos_data[90:100], 11.00))

    def test_transform_rows_boxcox04(self):
        # Prepare
        data, data_ref = blobs_data_generator01()

        # Run
        t_data = transform_rows_boxcox04(data)

        # Validate
        assert t_data is not None
        assert hasattr(t_data, 'shape')
        assert t_data.shape == data.shape

        assert not np.any(np.isnan(t_data))
        assert not np.any(np.isinf(t_data))

        # shift data to make it all positive
        pos_data = data + (-1 * data.min()) + 0.01

        assert np.all(t_data[0:20] == boxcox(pos_data[0:20], 0.00))
        assert np.all(t_data[20:40] == boxcox(pos_data[20:40], -0.25))
        assert np.all(t_data[40:60] == boxcox(pos_data[40:60], -0.50))
        assert np.all(t_data[60:80] == boxcox(pos_data[60:80], -0.75))
        assert np.all(t_data[80:100] == boxcox(pos_data[80:100], -1.00))

    def test_transform_rows_boxcox05(self):
        # Prepare
        data, data_ref = blobs_data_generator01()

        # Run
        t_data = transform_rows_boxcox05(data)

        # Validate
        assert t_data is not None
        assert hasattr(t_data, 'shape')
        assert t_data.shape == data.shape

        assert not np.any(np.isnan(t_data))
        assert not np.any(np.isinf(t_data))

        # shift data to make it all positive
        pos_data = data + (-1 * data.min()) + 0.01

        assert np.all(t_data[0:20] == boxcox(pos_data[0:20], -1.00))
        assert np.all(t_data[20:40] == boxcox(pos_data[20:40], -1.25))
        assert np.all(t_data[40:60] == boxcox(pos_data[40:60], -1.50))
        assert np.all(t_data[60:80] == boxcox(pos_data[60:80], -1.75))
        assert np.all(t_data[80:100] == boxcox(pos_data[80:100], -2.00))

    def test_transform_rows_boxcox06(self):
        # Prepare
        data, data_ref = blobs_data_generator01()

        # Run
        t_data = transform_rows_boxcox06(data)

        # Validate
        assert t_data is not None
        assert hasattr(t_data, 'shape')
        assert t_data.shape == data.shape

        assert not np.any(np.isnan(t_data))
        assert not np.any(np.isinf(t_data))

        # shift data to make it all positive
        pos_data = data + (-1 * data.min()) + 0.01

        assert np.all(t_data[0:10] == boxcox(pos_data[0:10], -2.00))
        assert np.all(t_data[10:20] == boxcox(pos_data[10:20], -3.00))
        assert np.all(t_data[20:30] == boxcox(pos_data[20:30], -4.00))
        assert np.all(t_data[30:40] == boxcox(pos_data[30:40], -5.00))
        assert np.all(t_data[40:50] == boxcox(pos_data[40:50], -6.00))
        assert np.all(t_data[50:60] == boxcox(pos_data[50:60], -7.00))
        assert np.all(t_data[60:70] == boxcox(pos_data[60:70], -8.00))
        assert np.all(t_data[70:80] == boxcox(pos_data[70:80], -9.00))
        assert np.all(t_data[80:90] == boxcox(pos_data[80:90], -10.00))
        assert np.all(t_data[90:100] == boxcox(pos_data[90:100], -11.00))

    def test_transform_rows_boxcox07(self):
        # Prepare
        data, data_ref = blobs_data_generator01()

        # Run
        t_data = transform_rows_boxcox07(data)

        # Validate
        assert t_data is not None
        assert hasattr(t_data, 'shape')
        assert t_data.shape == data.shape

        assert not np.any(np.isnan(t_data))
        assert not np.any(np.isinf(t_data))

        # shift data to make it all positive
        pos_data = data + (-1 * data.min()) + 0.01

        assert np.all(t_data[0:20] == boxcox(pos_data[0:20], -3.00))
        assert np.all(t_data[20:40] == boxcox(pos_data[20:40], -1.50))
        assert np.all(t_data[40:60] == boxcox(pos_data[40:60], 0.00))
        assert np.all(t_data[60:80] == boxcox(pos_data[60:80], 1.50))
        assert np.all(t_data[80:100] == boxcox(pos_data[80:100], 3.00))

    def test_transform_rows_boxcox_max_distance2(self):
        # Prepare
        data, data_ref = blobs_data_generator01()

        # Run
        t_data = transform_rows_boxcox_max_distance2(data)

        # Validate
        assert t_data is not None
        assert hasattr(t_data, 'shape')
        assert t_data.shape == data.shape

        assert not np.any(np.isnan(t_data))
        assert not np.any(np.isinf(t_data))

        # shift data to make it all positive
        pos_data = data + (-1 * data.min()) + 0.01

        assert np.all(t_data[0:20] == boxcox(pos_data[0:20], 0.00))
        assert np.all(t_data[20:40] == boxcox(pos_data[20:40], 0.50))
        assert np.all(t_data[40:60] == boxcox(pos_data[40:60], 1.00))
        assert np.all(t_data[60:80] == boxcox(pos_data[60:80], 1.50))
        assert np.all(t_data[80:100] == boxcox(pos_data[80:100], 2.00))

    def test_transform_rows_boxcox_max_distance12(self):
        # Prepare
        data, data_ref = blobs_data_generator01()

        # Run
        t_data = transform_rows_boxcox_max_distance12(data)

        # Validate
        assert t_data is not None
        assert hasattr(t_data, 'shape')
        assert t_data.shape == data.shape

        assert not np.any(np.isnan(t_data))
        assert not np.any(np.isinf(t_data))

        # shift data to make it all positive
        pos_data = data + (-1 * data.min()) + 0.01

        assert np.all(t_data[0:20] == boxcox(pos_data[0:20], -5.00))
        assert np.all(t_data[20:40] == boxcox(pos_data[20:40], -2.00))
        assert np.all(t_data[40:60] == boxcox(pos_data[40:60], 1.00))
        assert np.all(t_data[60:80] == boxcox(pos_data[60:80], 4.00))
        assert np.all(t_data[80:100] == boxcox(pos_data[80:100], 7.00))