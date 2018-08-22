import unittest
from unittest.mock import Mock
import numpy as np
from numbers import Number

from sklearn.metrics import adjusted_rand_score as ari

from experiments.execution import _run_experiment, _run_full_experiment


def _check_expected_noisy_data(data_returned, data_without_noise, noise_obj):
    percentage_objects = noise_obj['percentage_objects']
    noise_magnitude = noise_obj['magnitude']

    assert data_returned.shape == data_without_noise.shape

    n_objects = data_returned.shape[0]

    different_rows = []
    original_rows = []
    for row_idx in range(n_objects):
        returned_row = data_returned[row_idx]
        without_noise_row = data_without_noise[row_idx]

        if not np.array_equal(returned_row, without_noise_row):
            different_rows.append(returned_row)
            original_rows.append(without_noise_row)

    assert len(different_rows) == int(n_objects * percentage_objects)

    different_rows = np.array(different_rows)
    original_rows = np.array(original_rows)
    difference = np.abs(original_rows - different_rows)

    assert np.all(difference <= noise_magnitude)


class ExperimentExecutionTest(unittest.TestCase):
    def test_run_experiment_single_method(self):
        # Prepare
        data = np.random.rand(10, 5)
        data_ref = np.array([1, 2] * 5)
        data_n_clusters = 2

        data_generator01 = Mock(return_value=(data, data_ref))

        method01_return = np.array([1, 2] * 5)
        method01 = Mock(return_value=method01_return)
        method01.__doc__ = "   method01 doc   \n"

        # Run
        results = list(_run_experiment(
            0, data_generator01, methods=(method01,)
        ))

        # Validate
        data_generator01.assert_called_once_with(seed=None)

        assert method01.call_count == 1
        data_arg, n_clusters_arg = method01.call_args[0]
        assert np.array_equal(data, data_arg)
        assert data_n_clusters == n_clusters_arg

        assert results is not None
        assert len(results) == 1

        result_one = results[0]
        assert result_one is not None
        assert hasattr(result_one, '__iter__')
        assert len(result_one) == 3

        method_name, method_time, method_performance = result_one
        assert method_name is not None
        assert isinstance(method_name, str)
        assert method_name == 'method01 doc'

        assert method_time is not None
        assert isinstance(method_time, Number)

        assert method_performance is not None
        assert isinstance(method_performance, Number)
        assert method_performance == ari(data_ref, method01_return)

    def test_run_experiment_multiple_methods(self):
        # Prepare
        data = np.random.rand(10, 5)
        data_ref = np.array([1, 2] * 5)
        data_n_clusters = 2

        data_generator01 = Mock(return_value=(data, data_ref))

        method01_return = np.array([1, 2] * 5)
        method01 = Mock(return_value=method01_return)
        method01.__doc__ = "method01"

        method02_return = np.array([1, 2, 2, 2, 2] * 2)
        method02 = Mock(return_value=method02_return)
        method02.__doc__ = "method02"

        # Run
        results = list(_run_experiment(
            0,
            data_generator01,
            methods=(method01, method02)
        ))

        # Validate
        data_generator01.assert_called_once_with(seed=None)

        assert method01.call_count == 1
        data_arg, n_clusters_arg = method01.call_args[0]
        assert np.array_equal(data, data_arg)
        assert data_n_clusters == n_clusters_arg

        assert method02.call_count == 1
        data_arg, n_clusters_arg = method02.call_args[0]
        assert np.array_equal(data, data_arg)
        assert data_n_clusters == n_clusters_arg

        assert results is not None
        assert len(results) == 2

        one_result = results[0]
        method_name, method_time, method_performance = one_result
        assert method_name == 'method01'
        assert isinstance(method_time, Number)
        assert method_performance == ari(data_ref, method01_return)

        one_result = results[1]
        method_name, method_time, method_performance = one_result
        assert method_name == 'method02'
        assert isinstance(method_time, Number)
        assert method_performance == ari(data_ref, method02_return)

    def test_run_experiment_single_data_transform(self):
        # Prepare
        data = np.random.rand(10, 5)
        data_ref = np.array([1, 2] * 5)
        data_n_clusters = 2

        data_generator01 = Mock(return_value=(data, data_ref))

        data_transform01 = Mock(return_value=data + 5.0)

        method01_return = np.array([1, 2] * 5)
        method01 = Mock(return_value=method01_return)
        method01.__doc__ = "method01"

        method02_return = np.array([1, 2, 2, 2, 2] * 2)
        method02 = Mock(return_value=method02_return)
        method02.__doc__ = "method02"

        # Run
        results = list(_run_experiment(
            0,
            data_generator01,
            methods=(method01, method02),
            data_transform=data_transform01
        ))

        # Validate
        data_generator01.assert_called_once_with(seed=None)

        assert data_transform01.call_count == 1
        data_arg, = data_transform01.call_args[0]
        assert np.array_equal(data, data_arg)

        assert method01.call_count == 1
        data_arg, n_clusters_arg = method01.call_args[0]
        assert np.array_equal(data + 5.0, data_arg)
        assert data_n_clusters == n_clusters_arg

        assert method02.call_count == 1
        data_arg, n_clusters_arg = method02.call_args[0]
        assert np.array_equal(data + 5.0, data_arg)
        assert data_n_clusters == n_clusters_arg

        assert results is not None
        assert len(results) == 2

        one_result = results[0]
        method_name, method_time, method_performance = one_result
        assert method_name == 'method01'
        assert isinstance(method_time, Number)
        assert method_performance == ari(data_ref, method01_return)

        one_result = results[1]
        method_name, method_time, method_performance = one_result
        assert method_name == 'method02'
        assert isinstance(method_time, Number)
        assert method_performance == ari(data_ref, method02_return)

    @unittest.skip
    def test_run_experiment_with_noise_uniform01(self):
        # Prepare
        data = np.random.rand(10, 5)
        data_ref = np.array([1, 2] * 5)
        data_n_clusters = 2

        data_generator01 = Mock(return_value=(data.copy(), data_ref))

        data_transformed01 = data + 5.0
        data_transform01 = Mock(return_value=data_transformed01.copy())

        data_noise01 = {
            'percentage_objects': 0.3,
            'magnitude': 0.05,
        }

        method01_return = np.array([1, 2] * 5)
        method01 = Mock(return_value=method01_return)
        method01.__doc__ = "method01"

        method02_return = np.array([1, 2, 2, 2, 2] * 2)
        method02 = Mock(return_value=method02_return)
        method02.__doc__ = "method02"

        # Run
        results = list(_run_experiment(
            data_generator01,
            methods=(method01, method02),
            data_transform=data_transform01,
            data_noise=data_noise01,
        ))

        # Validate
        data_generator01.assert_called_once_with()

        assert data_transform01.call_count == 1
        data_arg, = data_transform01.call_args[0]
        assert np.array_equal(data, data_arg)

        assert method01.call_count == 1
        data_arg, n_clusters_arg = method01.call_args[0]
        _check_expected_noisy_data(data_arg, data_transformed01, data_noise01)
        assert data_n_clusters == n_clusters_arg

        assert method02.call_count == 1
        data_arg, n_clusters_arg = method02.call_args[0]
        _check_expected_noisy_data(data_arg, data_transformed01, data_noise01)
        assert data_n_clusters == n_clusters_arg

        assert results is not None
        assert len(results) == 2

        one_result = results[0]
        method_name, method_time, method_performance = one_result
        assert method_name == 'method01'
        assert isinstance(method_time, Number)
        assert method_performance == ari(data_ref, method01_return)

        one_result = results[1]
        method_name, method_time, method_performance = one_result
        assert method_name == 'method02'
        assert isinstance(method_time, Number)
        assert method_performance == ari(data_ref, method02_return)

    def test_run_full_experiment_always_same_return_value(self):
        # Prepare
        data = np.random.rand(10, 5)
        data_ref = np.array([1, 2] * 5)

        data_generator01 = Mock(return_value=(data.copy(), data_ref))

        data_transformed01 = data + 5.0
        data_transform01 = Mock(return_value=data_transformed01.copy())
        data_transform01.__name__ = 'data transform name'

        data_noise01 = {
            'percentage_objects': 0.3,
            'percentage_measures': 0.1,
            'magnitude': 0.05,
        }

        method01_return = np.array([1, 2] * 5)
        method01 = Mock(return_value=method01_return.copy())
        method01.__doc__ = "method01"

        method02_return = np.array([1, 2, 2, 2, 2] * 2)
        method02 = Mock(return_value=method02_return.copy())
        method02.__doc__ = "method02"

        experiment_data = {
            'n_reps': 5,
            'methods': (method01, method02),
            'data_generator': data_generator01,
            'data_transform': data_transform01,
            'data_noise': data_noise01,
        }

        # Run
        results = _run_full_experiment(experiment_data)

        # Validate
        assert data_generator01.call_count == 5
        assert data_transform01.call_count == 5
        assert method01.call_count == 5
        assert method02.call_count == 5

        assert results is not None
        assert hasattr(results, 'shape')
        assert results.shape == (10, 8)
        assert 'data_transf' in results.columns
        assert 'noise_perc_obj' in results.columns
        assert 'noise_perc_mes' in results.columns
        assert 'noise_mes_mag' in results.columns
        assert 'rep' in results.columns
        assert 'method' in results.columns
        assert 'time' in results.columns
        assert 'metric' in results.columns

        assert len(results['data_transf'].unique()) == 1
        assert 'data transform name' in results['data_transf'].unique()

        assert len(results['noise_mes_mag'].unique()) == 1
        assert 0.05 in results['noise_mes_mag'].unique()

        assert len(results['rep'].unique()) == 5
        assert 0 in results['rep'].unique()
        assert 1 in results['rep'].unique()
        assert 2 in results['rep'].unique()
        assert 3 in results['rep'].unique()
        assert 4 in results['rep'].unique()

        assert len(results['method'].unique()) == 2
        assert 'method01' in results['method'].unique()
        assert 'method02' in results['method'].unique()

        results_grp = results.groupby('method')['metric'].mean().round(3)
        assert results_grp.loc['method01'] == 1.00
        assert results_grp.loc['method02'] == -0.077

        results_grp = results.groupby('method')['time'].mean().round(3)
        assert results_grp.loc['method01'] < 1.0
        assert results_grp.loc['method02'] < 1.0

    def test_run_full_experiment_varying_return_value(self):
        # Prepare
        data = np.random.rand(10, 5)
        data_ref = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

        data_generator01 = Mock(return_value=(data.copy(), data_ref))

        data_transformed01 = data + 5.0
        data_transform01 = Mock(return_value=data_transformed01.copy())
        data_transform01.__name__ = 'data transform name'

        data_noise01 = {
            'percentage_objects': 0.3,
            'percentage_measures': 0.1,
            'magnitude': 0.05,
        }

        method01_return = np.array([
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],  # ari: 1.0
            [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],  # ari: 0.5970
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2],  # ari: 0.2941
            [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],  # ari: 1.0
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],  # ari: -0.0800
        ])
        method01 = Mock(side_effect=method01_return.copy())
        method01.__doc__ = "method01"

        method02_return = np.array([
            [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],  # ari: 1.0
            [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],  # ari: 0.5970
            [1, 1, 1, 1, 1, 1, 1, 1, 2, 2],  # ari: 0.09569
            [2, 2, 1, 1, 1, 1, 1, 1, 1, 1],  # ari: 0.09569
            [2, 2, 2, 2, 2, 1, 1, 2, 1, 2],  # ari: 0.2941
        ])
        method02 = Mock(side_effect=method02_return.copy())
        method02.__doc__ = "method02"

        experiment_data = {
            'n_reps': 5,
            'methods': (method01, method02),
            'data_generator': data_generator01,
            'data_transform': data_transform01,
            'data_noise': data_noise01,
        }

        # Run
        results = _run_full_experiment(experiment_data)

        # Validate
        assert data_generator01.call_count == 5
        assert data_transform01.call_count == 5
        assert method01.call_count == 5
        assert method02.call_count == 5

        assert results is not None
        assert hasattr(results, 'shape')
        assert results.shape == (10, 8), results.shape

        assert len(results['method'].unique()) == 2
        assert 'method01' in results['method'].unique()
        assert 'method02' in results['method'].unique()

        results_grp = results.groupby('method')['metric'].mean().round(2)
        assert results_grp.loc['method01'] == 0.56
        assert results_grp.loc['method02'] == 0.42

        results_grp = results.groupby('method')['time'].mean().round(2)
        assert results_grp.loc['method01'] < 1.0
        assert results_grp.loc['method02'] < 1.0
