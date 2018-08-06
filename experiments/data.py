from functools import partial

import numpy as np
from scipy.stats import boxcox
from sklearn.datasets import make_blobs
from sklearn.preprocessing import minmax_scale

from clustermatch.cluster import run_quantile_clustering


def blobs_data_generator01():
    """
    Blobs. n_samples=100, n_features=20, centers=3, cluster_std=0.10, center_box=(-1.0, 1.0)
    """
    return make_blobs(
        n_samples=100,
        centers=3,
        n_features=20,
        cluster_std=0.10,
        shuffle=True,
        center_box=(-1.0, 1.0)
    )


def blobs_data_generator02(seed=None, n_samples=100, n_features=1000):
    """
    Blobs. n_samples=100, n_features=1000, centers=3, cluster_std=0.10, center_box=(-1.0, 1.0)
    """
    return make_blobs(
        n_samples=n_samples,
        centers=3,
        n_features=n_features,
        cluster_std=0.10,
        shuffle=True,
        center_box=(-1.0, 1.0),
        random_state=seed,
    )


def _get_array_chunks(data, chunk_size):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(data), chunk_size):
        sl = slice(i, i + chunk_size)
        yield sl, data[sl]


def _apply_noise(data, data_noise):
    data_n_objects = data.shape[1]
    data_n_measures = data.shape[0]

    if len(data_noise) == 0:
        return data

    percentage_objects = data_noise.get('percentage_objects', 0.1)
    percentage_measures = data_noise.get('percentage_measures', 0.0)
    magnitude = data_noise.get('magnitude', 0.0)

    selected_rows = np.random.choice(
        data_n_measures,
        size=int(data_n_measures * percentage_measures),
        replace=False
    )

    selected_cols = np.random.choice(
        data_n_objects,
        size=int(data_n_objects * percentage_objects),
        replace=False
    )

    noisy_data = data.copy()

    if np.issubdtype(data.dtype, np.number) or all([np.isreal(x) for row in data for x in row]):
        if not np.issubdtype(data.dtype, np.number):
            data = data.astype(float)

        if len(selected_rows) > 0:
            noisy_points = np.random.rand(len(selected_rows), data_n_objects)
            noisy_points = minmax_scale(noisy_points, axis=1, feature_range=(data.min(), data.max()))
            noisy_points = noisy_points * magnitude

            noisy_data[selected_rows, :] += noisy_points

        if len(selected_cols) > 0:
            noisy_points = np.random.rand(data_n_measures, len(selected_cols))
            noisy_points = minmax_scale(noisy_points, axis=1, feature_range=(data.min(), data.max()))

            noisy_data[:, selected_cols] = noisy_points

    else:
        assert all([not np.isreal(x) for row in data for x in row])

        unique_cat = np.unique(data)

        if len(selected_cols) > 0:
            # noisy_points = np.random.rand(data_n_measures, len(selected_cols))
            noisy_points = np.random.choice(unique_cat, (data_n_measures, len(selected_cols)))
            # noisy_points = minmax_scale(noisy_points, axis=1, feature_range=(data.min(), data.max()))

            noisy_data[:, selected_cols] = noisy_points

        # for i in range(data.shape[0]):
        #     for j in range(data.shape[1]):
        #         if np.random.rand() < magnitude:
        #             noisy_data[i, j] = np.random.choice(unique_cat)

    return noisy_data


def _generic_data_transformation(data, sources_transformers, dtype=None, **kwargs):
    if len(sources_transformers) == 0:
        return data

    n_data = data.shape[0]
    n_sim_sources = len(sources_transformers)
    data_step = int(n_data / n_sim_sources)

    t_data = np.empty(data.shape, dtype=data.dtype if dtype is None else dtype)
    i = 0

    for sl, data_chunk in _get_array_chunks(data, data_step):
        transformer = sources_transformers[i % n_sim_sources]

        # transform
        if callable(transformer):
            t_data_chunk = transformer(data_chunk)
        else:
            t_data_chunk = data_chunk * transformer

        t_data[sl] = t_data_chunk

        # if not np.issubdtype(t_data_chunk.dtype, np.number):
        #     is_data_object = True

        # data noise
        if 'data_noise' in kwargs:
            data_noise = kwargs['data_noise']
            t_data[sl] = _apply_noise(t_data[sl], data_noise)

        i += 1

    return t_data


def transform_rows_nonlinear_and_categorical01(data, **kwargs):
    """
    Nonlinear and categorical row transformation 01. 7 numerical data sources (x^4, log, exp2, 100, x^5, 10000, 0.0001) and 3 categorical (10, 4 and 2 categories).
    """
    def create_categorical(data, cats):
        n_cats = len(cats)
        t_data = np.empty(data.shape, dtype=object)

        for data_row_idx, data_row in enumerate(data):
            data_row_part = run_quantile_clustering(data_row, n_cats)
            t_data[data_row_idx] = np.array([cats[int(x)] for x in data_row_part])

        return t_data

    sources_transformers = [
        lambda x: np.power(x, 4),
        lambda x: np.log(np.abs(x)),
        lambda x: np.exp2(x),
        100.0,
        lambda x: create_categorical(x, cats=[
            'cat01', 'cat02', 'cat03', 'cat04',
            'cat05', 'cat06', 'cat07', 'cat08',
            'cat09', 'cat10',
        ]),
        lambda x: np.power(x, 5),
        10000.0,
        lambda x: create_categorical(x, cats=['cat01', 'cat02', 'cat03', 'cat04']),
        0.0001,
        lambda x: create_categorical(x, cats=['cat01', 'cat02']),
    ]

    return _generic_data_transformation(data, sources_transformers, dtype=object, **kwargs)


def transform_rows_full_scaled01(data):
    """
    Full row scale. 5 simulated data sources; values: 0.01, 0.1, 10, 100, 1000
    """
    sources_transformers = [0.01, 0.1, 10.0, 100.0, 1000.0]

    return _generic_data_transformation(data, sources_transformers)


def transform_rows_nonlinear01(data, **kwargs):
    """
    Nonlinear row transformation 01. 5 simulated data sources; Functions: exp, x^2, log, expm1, log10
    """
    sources_transformers = [
        np.exp,
        lambda x: np.power(x, 2),
        lambda x: np.log(np.abs(x)),
        np.expm1,
        lambda x: np.log10(np.abs(x)),
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear02(data, **kwargs):
    """
    Nonlinear row transformation 02. 4 simulated data sources; Functions: x^3, log, log1p, exp2
    """
    sources_transformers = [
        lambda x: np.power(x, 3),
        lambda x: np.log(np.abs(x)),
        lambda x: np.log1p(np.abs(x)),
        np.exp2,
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear03(data, **kwargs):
    """
    Nonlinear row transformation 03. 10 simulated data sources; Functions: x^4, log, exp2, 100, log1p, x^5, 10000, log10, 0.0001, log2
    """
    sources_transformers = [
        lambda x: np.power(x, 4),
        lambda x: np.log(np.abs(x)),
        lambda x: np.exp2(x),
        100.0,
        lambda x: np.log1p(np.abs(x)),
        lambda x: np.power(x, 5),
        10000.0,
        lambda x: np.log10(np.abs(x)),
        0.0001,
        lambda x: np.log2(np.abs(x)),
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear03_01(data, **kwargs):
    """
    Nonlinear row transformation 03_01. 10 simulated data sources; Functions: x^2, log, exp2, 100, log1p, x^3, 10000, log10, 0.0001, log2
    """
    sources_transformers = [
        lambda x: np.power(x, 2),
        lambda x: np.log(np.abs(x)),
        lambda x: np.exp2(x),
        100.0,
        lambda x: np.log1p(np.abs(x)),
        lambda x: np.power(x, 3),
        10000.0,
        lambda x: np.log10(np.abs(x)),
        0.0001,
        lambda x: np.log2(np.abs(x)),
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear04(data, **kwargs):
    """
    Nonlinear row transformation 04. 10 simulated data sources; Functions: 1.0, 0.5*(x+1)^2, sin(pi*x), cos(pi*x), x^5, exp2, log10, boxcox(2), boxcox(4), boxcox(6).
    """
    sources_transformers = [
        1.0,
        lambda x: 0.5 * np.power((x+1), 2),
        lambda x: np.sin(np.pi * x),
        lambda x: np.cos(np.pi * x),
        lambda x: np.power(x, 5),
        lambda x: np.exp2(x),
        lambda x: np.log10(np.abs(x)),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 2.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 4.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 6.00),
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear05(data, **kwargs):
    """
    Nonlinear row transformation 05. 10 simulated data sources; Functions: 1.0, 0.5*(x+1)^2, sin(pi*x), cos(pi*x), x^5, exp2, log10(x-x.min()), boxcox(2), boxcox(4), boxcox(6).
    """
    sources_transformers = [
        1.0,
        lambda x: 0.5 * np.power((x+1), 2),
        lambda x: np.sin(np.pi * x),
        lambda x: np.cos(np.pi * x),
        lambda x: np.power(x, 5),
        lambda x: np.exp2(x),
        lambda x: np.log10(x + (-1.0 * x.min()) + 0.01),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 2.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 4.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 6.00),
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear06(data, **kwargs):
    """
    Nonlinear row transformation 06. 12 simulated data sources; Functions: 1.0, 0.5*(x+1)^2, sin(pi*x), sin(2*pi*x), cos(pi*x), cos(2*pi*x), x^5, exp2, log10(x-x.min()), boxcox(2), boxcox(4), boxcox(6).
    """
    sources_transformers = [
        1.0,
        lambda x: 0.5 * np.power((x+1), 2),
        lambda x: np.sin(np.pi * x),
        lambda x: np.sin(2.0 * np.pi * x),
        lambda x: np.cos(np.pi * x),
        lambda x: np.cos(2.0 * np.pi * x),
        lambda x: np.power(x, 5),
        lambda x: np.exp2(x),
        lambda x: np.log10(x + (-1.0 * x.min()) + 0.01),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 2.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 4.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 6.00),
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear07(data, **kwargs):
    """
    Nonlinear row transformation 07. 12 simulated data sources; Functions: 1.0, 0.5*(x+1)^2, sin(pi*x), -100, cos(pi*x), 0.0001, x^5, exp2, log10(x-x.min()), boxcox(2), boxcox(4), boxcox(6).
    """
    sources_transformers = [
        1.0,
        lambda x: 0.5 * np.power((x+1), 2),
        lambda x: np.sin(np.pi * x),
        -100.0,
        lambda x: np.cos(np.pi * x),
        0.0001,
        lambda x: np.power(x, 5),
        lambda x: np.exp2(x),
        lambda x: np.log10(x + (-1.0 * x.min()) + 0.01),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 2.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 4.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 6.00),
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear08(data, **kwargs):
    """
    Nonlinear row transformation 08. 5 simulated data sources; Functions: boxcox(0), boxcox(1), boxcox(2), boxcox(3), boxcox(4).
    """
    sources_transformers = [
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 0.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 1.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 2.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 3.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 4.00),
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear09(data, **kwargs):
    """
    Nonlinear row transformation 09. 5 simulated data sources; Functions: x^2, boxcox(1), boxcox(2), boxcox(3), boxcox(4).
    """
    sources_transformers = [
        lambda x: np.power(x, 2),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 1.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 2.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 3.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 4.00),
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear10(data, **kwargs):
    """
    Nonlinear row transformation 10. 5 simulated data sources; Functions: x^2, log(x), boxcox(2), boxcox(3), boxcox(4).
    """
    sources_transformers = [
        lambda x: np.power(x, 2),
        lambda x: np.log(np.abs(x)),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 2.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 3.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 4.00),
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear11(data, **kwargs):
    """
    Nonlinear row transformation 11. 5 simulated data sources; Functions: x^2, log(x), x^4, boxcox(3), boxcox(4).
    """
    sources_transformers = [
        lambda x: np.power(x, 2),
        lambda x: np.log(np.abs(x)),
        lambda x: np.power(x, 4),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 3.00),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 4.00),
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear12(data, **kwargs):
    """
    Nonlinear row transformation 12. 5 simulated data sources; Functions: x^2, log(x), x^4, sin(pi * x), boxcox(4).
    """
    sources_transformers = [
        lambda x: np.power(x, 2),
        lambda x: np.log(np.abs(x)),
        lambda x: np.power(x, 4),
        lambda x: np.sin(np.pi * x),
        lambda x: boxcox(x + (-1.0 * x.min()) + 0.01, 4.00),
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear13(data, **kwargs):
    """
    Nonlinear row transformation 13. 5 simulated data sources; 1, 1e-10, 1e10, -1e-10, -1e10.
    """
    sources_transformers = [
        1,
        1e-10,
        1e10,
        -1e-10,
        -1e10,
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear14(data, **kwargs):
    """
    Nonlinear row transformation 14. 5 simulated data sources; x^2, 1e-10, 1e10, -1e-10, -1e10.
    """
    sources_transformers = [
        lambda x: np.power(x, 2),
        1e-10,
        1e10,
        -1e-10,
        -1e10,
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear15(data, **kwargs):
    """
    Nonlinear row transformation 15. 5 simulated data sources; x^2, log(abs(x)), 1e10, -1e-10, -1e10.
    """
    sources_transformers = [
        lambda x: np.power(x, 2),
        lambda x: np.log(np.abs(x)),
        1e10,
        -1e-10,
        -1e10,
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear16(data, **kwargs):
    """
    Nonlinear row transformation 16. 5 simulated data sources; x^2, log(abs(x)), x^4, -1e-10, -1e10.
    """
    sources_transformers = [
        lambda x: np.power(x, 2),
        lambda x: np.log(np.abs(x)),
        lambda x: np.power(x, 4),
        -1e-10,
        -1e10,
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def transform_rows_nonlinear17(data, **kwargs):
    """
    Nonlinear row transformation 17. 5 simulated data sources; x^2, log(abs(x)), x^4, sin(pi*x), -1e10.
    """
    sources_transformers = [
        lambda x: np.power(x, 2),
        lambda x: np.log(np.abs(x)),
        lambda x: np.power(x, 4),
        lambda x: np.sin(np.pi * x),
        -1e10,
    ]

    return _generic_data_transformation(data, sources_transformers, **kwargs)


def _boxcox_data_transformation(data, sources_transformers):
    # make sure all data is positive
    final_data = data.copy()

    if np.any(final_data < 0):
        final_data = data + (-1 * data.min()) + 0.01

    return _generic_data_transformation(final_data, sources_transformers)


def transform_rows_boxcox01(data):
    """
    BoxCox row transformation 01. 5 simulated data sources; Lambdas from 0.0 to 1.0 (0.00, 0.25, 0.50, 0.75, 1.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(0.00, 1.00, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox02(data):
    """
    BoxCox row transformation 02. 5 simulated data sources; Lambdas from 1.0 to 2.0 (1.00, 1.25, 1.50, 1.75, 2.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(1.00, 2.00, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox03(data):
    """
    BoxCox row transformation 03. 10 simulated data sources; Lambdas from 2.0 to 11.0 (2.00, 3.00, ..., 10.0, 11.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(2.00, 11.00, num=10)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox04(data):
    """
    BoxCox row transformation 04. 5 simulated data sources; Lambdas from 0.0 to -1.0 (0.00, -0.25, -0.50, -0.75, -1.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(0.00, -1.00, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox05(data):
    """
    BoxCox row transformation 05. 5 simulated data sources; Lambdas from -1.0 to -2.0 (-1.00, -1.25, -1.50, -1.75, -2.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-1.00, -2.00, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox06(data):
    """
    BoxCox row transformation 06. 10 simulated data sources; Lambdas from -2.0 to -11.0 (-2.00, -3.00, ..., -10.0, -11.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-2.00, -11.00, num=10)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox07(data):
    """
    BoxCox row transformation 07. 5 simulated data sources; Lambdas from -3.0 to 3.0 (-3.00, -1.50, 0.00, 1.50, 3.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-3.00, 3.00, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance2(data):
    """
    BoxCox row transformation, maximum lambda distance: 2. 5 simulated data sources; Lambdas from 0.0 to 2.0 (0.00,  0.50,  1.00,  1.50,  2.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(0.00, 2.00, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance3(data):
    """
    BoxCox row transformation, maximum lambda distance: 3. 5 simulated data sources; Lambdas from -0.50 to 2.5 (-0.50, 0.25, 1.00, 1.75, 2.50)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-0.50, 2.50, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance4(data):
    """
    BoxCox row transformation, maximum lambda distance: 4. 5 simulated data sources; Lambdas from -1.00 to 3.0 (-1.00,  0.00,  1.00,  2.00,  3.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-1.00, 3.00, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance5(data):
    """
    BoxCox row transformation, maximum lambda distance: 5. 5 simulated data sources; Lambdas from -1.50 to 3.5 (-1.50, -0.25,  1.00,  2.25,  3.50)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-1.50, 3.50, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance6(data):
    """
    BoxCox row transformation, maximum lambda distance: 6. 5 simulated data sources; Lambdas from -2.00 to 4.0 (-2.00, -0.50,  1.00,  2.50,  4.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-2.00, 4.00, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance7(data):
    """
    BoxCox row transformation, maximum lambda distance: 7. 5 simulated data sources; Lambdas from -2.50 to 4.50 (-2.50, -0.75,  1.00,  2.75,  4.50)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-2.50, 4.50, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance8(data):
    """
    BoxCox row transformation, maximum lambda distance: 8. 5 simulated data sources; Lambdas from -3.00 to 5.00 (-3.00, -1.00,  1.00,  3.00,  5.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-3.00, 5.00, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance9(data):
    """
    BoxCox row transformation, maximum lambda distance: 9. 5 simulated data sources; Lambdas from -3.50 to 5.50 (-3.50, -1.25,  1.00,  3.25,  5.50)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-3.50, 5.50, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance10(data):
    """
    BoxCox row transformation, maximum lambda distance: 10. 5 simulated data sources; Lambdas from -4.00 to 6.00 (-4.00, -1.50,  1.00,  3.50,  6.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-4.00, 6.00, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance11(data):
    """
    BoxCox row transformation, maximum lambda distance: 11. 5 simulated data sources; Lambdas from -4.50 to 6.50 (-4.50, -1.75,  1.00,  3.75,  6.50)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-4.50, 6.50, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance12(data):
    """
    BoxCox row transformation, maximum lambda distance: 12. 5 simulated data sources; Lambdas from -5.00 to 7.00 (-5.00, -2.00,  1.00,  4.00,  7.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-5.00, 7.00, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance13(data):
    """
    BoxCox row transformation, maximum lambda distance: 13. 5 simulated data sources; Lambdas from -5.50 to 7.50 (-5.50, -2.25,  1.00,  4.25,  7.50)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-5.50, 7.50, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance14(data):
    """
    BoxCox row transformation, maximum lambda distance: 14. 5 simulated data sources; Lambdas from -6.00 to 8.00 (-6.00, -2.50,  1.00,  4.50,  8.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-6.00, 8.00, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance15(data):
    """
    BoxCox row transformation, maximum lambda distance: 15. 5 simulated data sources; Lambdas from -6.50 to 8.50 (-6.50, -2.75,  1.00,  4.75,  8.50)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-6.50, 8.50, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance16(data):
    """
    BoxCox row transformation, maximum lambda distance: 16. 5 simulated data sources; Lambdas from -7.00 to 9.00 (-7.00, -3.00,  1.00,  5.00,  9.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-7.00, 9.00, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance17(data):
    """
    BoxCox row transformation, maximum lambda distance: 17. 5 simulated data sources; Lambdas from -7.50 to 9.50 (-7.50, -3.25,  1.00,  5.25,  9.50)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-7.50, 9.50, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance18(data):
    """
    BoxCox row transformation, maximum lambda distance: 18. 5 simulated data sources; Lambdas from -8.00 to 10.00 (-8.00,  -3.50,   1.00,   5.50,  10.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-8.00, 10.00, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance19(data):
    """
    BoxCox row transformation, maximum lambda distance: 19. 5 simulated data sources; Lambdas from -8.50 to 10.50 (-8.50,  -3.75,   1.00,   5.75,  10.50)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-8.50, 10.50, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)


def transform_rows_boxcox_max_distance20(data):
    """
    BoxCox row transformation, maximum lambda distance: 20. 5 simulated data sources; Lambdas from -9.00 to 11.00 (-9.00,  -4.00,   1.00,   6.00,  11.00)
    """
    sources_transformers = [partial(lambda x, a: boxcox(x, a), a=alpha) for alpha in np.linspace(-9.00, 11.00, num=5)]

    return _boxcox_data_transformation(data, sources_transformers)
