from multiprocessing import cpu_count

import numpy as np
from scipy import stats
from scipy.signal import argrelextrema
from scipy.spatial.distance import squareform, cdist
from scipy.cluster.hierarchy import fcluster, linkage
import pandas as pd
from joblib import Parallel, delayed

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import scale


def _get_perc_from_k(k):
    return [(1.0 / k) * i for i in range(1, k)]


def run_quantile_clustering(data, k, **kwargs):
    data_perc = stats.rankdata(data, 'average') / len(data)
    data_perc_sort_idx = np.argsort(data_perc)

    # data_perc = data
    percentiles = [0.0] + _get_perc_from_k(k) + [1.0]

    cut_points = np.searchsorted(data_perc[data_perc_sort_idx], percentiles, side='right')

    current_cluster = 0
    part = np.zeros(data.shape, dtype=float) - 1

    for i in range(len(cut_points) - 1):
        lim1 = cut_points[i]
        lim2 = cut_points[i+1]

        part[data_perc_sort_idx[lim1:lim2]] = current_cluster
        current_cluster += 1

    return part


def categorical_clustering(data, **kwargs):
    cat_clus = {v: k for k, v in enumerate(np.unique(data))}

    return np.array([cat_clus[x] for x in data])


def run_maxdiff_clustering(data, k, **kwargs):
    data_idx_sorted = np.argsort(data)
    data_sorted = data[data_idx_sorted]

    data_diffs = np.array([abs(data_sorted[i] - data_sorted[i+1]) for i in range(len(data_sorted)-1)])
    data_diffs_idx_sort = np.argsort(data_diffs)[::-1]

    break_points = data_diffs_idx_sort[:(k-1)]
    break_points = np.sort(break_points)
    break_points = break_points[::-1]

    partition = np.zeros(len(data), int)
    current_cluster = 1

    for idx in range(len(break_points)):
        curr_bp_idx = break_points[idx] + 1

        partition[data_idx_sorted[:curr_bp_idx]] = current_cluster
        current_cluster += 1

    return partition


def run_kde_clustering(data, k, bandw=1.0, **kwargs):
    # FIXME: the next line was commented out in patch
    data = scale(data)
    data_res = data.reshape((-1,))

    mi = []
    mi_counter = 0

    while len(mi) + 1 < k and mi_counter < 3:
        bandw = bandw / 1.8
        s, kde_sample = _run_kde(data, bandw)

        mi = argrelextrema(kde_sample, np.less)[0]

        mi_counter += 1

    if mi_counter >= 3:
        raise Exception('bandwidth did not produced at least {0} clusters'.format(k))

    samples_idx_sorted = np.argsort(kde_sample[mi])
    samples_idx_sorted = mi[samples_idx_sorted]

    partition = np.zeros(len(data), int)

    current_cluster = 1
    for idx in range(k - 1):
        mi_idx = samples_idx_sorted[idx]
        mi_idx_value = s[mi_idx]

        if idx == 0:
            partition[data_res < mi_idx_value] = current_cluster
            current_cluster += 1

        if idx != (k - 1 - 1):
            mi_next_idx = samples_idx_sorted[idx + 1]
            mi_next_idx_value = s[mi_next_idx]

            if mi_idx_value < mi_next_idx_value:
                partition[(data_res >= mi_idx_value) * (data_res < mi_next_idx_value)] = current_cluster
            else:
                partition[(data_res >= mi_next_idx_value) * (data_res < mi_idx_value)] = current_cluster

            current_cluster += 1
        # else:
        #     partition[(data_res >= mi_idx_value)] = current_cluster

    return partition


def _run_kde(data, bandwidth):
    data_std = data.std()

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data.reshape(-1, 1))
    s = np.linspace(data.min() - data_std, data.max() + data_std)
    e = kde.score_samples(s.reshape(-1, 1))

    return s, e


def get_squareform(similarity_pdist, fill_diag_value=1.0):
    assert similarity_pdist is not None, 'similarity_pdist is None'
    assert hasattr(similarity_pdist, 'shape'), 'similarity_pdist has no shape attribute'
    assert len(similarity_pdist.shape) == 1, 'similarity_pdist has incorrect shape'

    similarity_square = squareform(similarity_pdist.astype(float))
    np.fill_diagonal(similarity_square, fill_diag_value)

    return similarity_square


def _isempty(row):
    return np.array([x is None or (np.isreal(x) and np.isnan(x)) for x in row])


def _get_common_features(obj1, obj2):
    obj1_notnan = np.logical_not(_isempty(obj1))
    obj2_notnan = np.logical_not(_isempty(obj2))

    common_features = np.logical_and(obj1_notnan, obj2_notnan)
    n_common_features = common_features.sum()

    return common_features, n_common_features


def _get_range_n_clusters(n_common_features, **kwargs):
    internal_n_clusters = kwargs.get('internal_n_clusters')

    if internal_n_clusters is None:
        estimated_k = int(np.floor(np.sqrt(n_common_features)))
        estimated_k = np.min((estimated_k, 10))
        range_n_clusters = range(2, np.max((estimated_k, 3)))
    elif isinstance(internal_n_clusters, (tuple, list, range)):
        range_n_clusters = internal_n_clusters
    elif isinstance(internal_n_clusters, int):
        range_n_clusters = (internal_n_clusters,)
    else:
        raise ValueError('n_clusters is invalid')

    return range_n_clusters


def _get_internal_parts(data_obj, range_n_clusters, clustering_method, **kwargs):
    partitions = []

    for k in range_n_clusters:
        if len(data_obj) <= k:
            part = np.array([np.nan] * len(data_obj))
        else:
            if all([np.isreal(x) for x in data_obj]):
                part = clustering_method(data_obj, k, **kwargs)
                # part_k = len(np.unique(part))
                # assert part_k == k, 'partition has the wrong number of clusters: {}'.format(part_k)
            elif all([not np.isreal(x) for x in data_obj]):
                part = categorical_clustering(data_obj, **kwargs)
            else:
                raise ValueError('Data is not all numerical or not all categorical')

        partitions.append(part)

    return np.array(partitions)


def _get_clustering_method(**kwargs):
    internal_clustering_method = kwargs.get('internal_clustering_method', 'quantiles')

    if internal_clustering_method == 'kmeans':
        return run_kmeans
    elif internal_clustering_method == 'quantiles':
        return run_quantile_clustering
    else:
        raise ValueError('invalid internal clustering method')


def row_col_from_condensed_index(d,i):
    b = 1 -2*d
    x = np.floor((-b - np.sqrt(b**2 - 8*i))/2)
    y = i + x*(b + x + 2)/2 + 1
    return (int(x), int(y))


def _compute_ari(part1, part2):
    if np.isnan(part1).any() or len(part1) == 0:
        return 0.0

    return ari(part1, part2)


def _get_contingency_table(p1, p2):
    df = pd.DataFrame({'p1': p1, 'p2': p2})
    return pd.crosstab(df.p1, df.p2).values


def _get_fisher_pvalue(p1, p2):
    raise NotImplemented()
    # contingency_table = _get_contingency_table(p1, p2)
    #return fisher_exact(contingency_table)


def _get_chi2(p1, p2):
    # n_objects = len(p1)

    p1_clusters = np.unique(p1)
    p2_clusters = np.unique(p2)

    if len(p1_clusters) == 1 or len(p2_clusters) == 1:
        print('Partition with one cluster. Skipping')
        return np.nan, 1.0

    contingency_table = _get_contingency_table(p1, p2)

    chi2, chi2_pvalue, dof, expected = stats.chi2_contingency(contingency_table)

    # k, l = len(p1_clusters), len(p2_clusters)
    # vmes = np.sqrt(chi2 / (n_objects * np.min((k - 1, l - 1))))

    return chi2_pvalue


def _get_perm_set(obj1, obj2, clustering_method, perm_range):
    rs = np.random.RandomState(perm_range[0])
    # print(perm_range[0])

    other_cms = []
    for i in perm_range:
        obj2_permuted = rs.permutation(obj2)

        new_cm = cm(obj1, obj2_permuted, clustering_method=clustering_method)[0]
        other_cms.append(new_cm)

    return other_cms


def _get_perm_pvalue(orig_cm, obj1, obj2, clustering_method, n_perm, perm_n_jobs=1, **kwargs):
    n_cpus = perm_n_jobs if perm_n_jobs > 0 else cpu_count()
    step = int(np.ceil(n_perm / n_cpus))
    full_perm_range = range(0, n_perm, step)

    other_cms = Parallel(n_jobs=perm_n_jobs)(
        delayed(_get_perm_set)(obj1, obj2, clustering_method, perm_range)
        for perm_range in [range(s, min(s + full_perm_range.step, full_perm_range.stop)) for s in full_perm_range]
    )

    # flatten
    other_cms = np.array([cm_val for sublist in other_cms for cm_val in sublist])

    return (len(other_cms[other_cms >= orig_cm]) + 1) / (len(other_cms) + 1)


def cm(obj1, obj2, clustering_method=run_quantile_clustering, compute_perm_pvalue=False,
       compute_chi2_pvalue=False, compute_fisher_pvalue=False, **kwargs):
    range_n_clusters = _get_range_n_clusters(len(obj1), **kwargs)

    obj1_parts = _get_internal_parts(obj1, range_n_clusters, clustering_method, **kwargs)
    obj2_parts = _get_internal_parts(obj2, range_n_clusters, clustering_method, **kwargs)

    comp_values = cdist(obj1_parts, obj2_parts, metric=_compute_ari)

    max_pos = np.unravel_index(comp_values.argmax(), comp_values.shape)
    max_ari = comp_values[max_pos]

    obj1_max_part = obj1_parts[max_pos[0]]
    obj2_max_part = obj2_parts[max_pos[1]]

    if len(np.unique(obj1_max_part)) == 1 or len(np.unique(obj2_max_part)) == 1:
        return 0.0, 1.0

    sim_value = max_ari
    pvalue = None if not compute_fisher_pvalue else _get_fisher_pvalue(obj1_max_part, obj2_max_part)

    if compute_chi2_pvalue and not compute_fisher_pvalue:
        pvalue = _get_chi2(obj1_max_part, obj2_max_part)

    if compute_perm_pvalue:
        pvalue = _get_perm_pvalue(sim_value, obj1, obj2, clustering_method, **kwargs)

    return sim_value, pvalue


def get_shared_objects(obj1, obj2, **kwargs):
    n_common_features = _get_common_features(obj1, obj2)[1]
    return (n_common_features,)


def get_pearson(obj1, obj2, **kwargs):
    if len(np.unique(obj1)) == 1 or len(np.unique(obj2)) == 1 or \
            any([not np.isreal(x) for x in obj1]) or \
            any([not np.isreal(x) for x in obj2]):
        return 0.0, 1.0

    return stats.pearsonr(obj1, obj2)


def _calculate_sub_simmatrix(data, idx_range, sim_func='cm', return_pvalue=False, min_n_common_features=3, **kwargs):
    p_dist = []
    p_dist_pvalue = []
    n_objects = data.shape[0]

    if sim_func == 'cm':
        similarity_func = cm
    elif sim_func == 'pearson':
        similarity_func = get_pearson
    elif sim_func == 'shared_objects':
        similarity_func = get_shared_objects
    else:
        raise ValueError('Invalid sim_func')

    for idx in idx_range:
        obj1_idx, obj2_idx = row_col_from_condensed_index(n_objects, idx)

        obj1 = data[obj1_idx]
        obj2 = data[obj2_idx]

        common_features, n_common_features = _get_common_features(obj1, obj2)

        if n_common_features < min_n_common_features:
            sim_values = (0.0, 1.0) # sim value and pvalue
        else:
            sim_values = similarity_func(obj1[common_features], obj2[common_features], **kwargs)

        p_dist.append(sim_values[0])

        if return_pvalue:
            p_dist_pvalue.append(sim_values[1])

    return p_dist, p_dist_pvalue


def calculate_simmatrix(data, fill_diag_value=1.0, n_jobs=1, **kwargs):
    data_index = None
    if hasattr(data, 'index'):
        data_index = data.index.tolist()

    if hasattr(data, 'values'):
        data = data.values

    # FIXME: quantiles clustering is only for 1d comparisons. Use kmeans for n-dimensional.
    kwargs['clustering_method'] = _get_clustering_method(**kwargs)

    n_objects = data.shape[0]

    p_dist_len = int((n_objects * (n_objects - 1)) / 2)

    # FIXME: set n_jobs according to data size. Do some performance test with unit tests
    n_cpus = n_jobs if n_jobs > 0 else cpu_count()

    step = int(np.ceil(p_dist_len / n_cpus))
    p_dist_range = range(0, p_dist_len, step)

    p_dist_values = Parallel(n_jobs=n_jobs)(
        delayed(_calculate_sub_simmatrix)(data, idx_range, **kwargs)
        for idx_range in [range(s, min(s + p_dist_range.step, p_dist_range.stop)) for s in p_dist_range]
    )

    p_dist = []
    p_dist_pval = []
    for p, pval in p_dist_values:
        p_dist.extend(p)

        if len(pval) > 0:
            p_dist_pval.extend(pval)

    p_dist = np.array(p_dist)
    p_dist_pval = np.array(p_dist_pval)

    return_pvalue = kwargs.get('return_pvalue', False)

    if data_index is not None:
        sqmatrix = get_squareform(p_dist, fill_diag_value)
        sim_matrix = pd.DataFrame(
            sqmatrix,
            index=data_index,
            columns=data_index
        )

        if return_pvalue:
            sqmatrix_pval = get_squareform(p_dist_pval, np.nan)
            pval_matrix = pd.DataFrame(
                sqmatrix_pval,
                index=data_index,
                columns=data_index
            )
    else:
        sim_matrix = p_dist
        pval_matrix = p_dist_pval

    if return_pvalue:
        return sim_matrix, pval_matrix
    else:
        return sim_matrix


def _get_partition(clustering_method, n_clusters, sim_matrix):
    # get all partitions
    partitions = {}
    for k in n_clusters:
        column_name = 'k={0}'.format(str(k))
        partitions[column_name] = clustering_method(k)

    final_partition = pd.DataFrame(partitions)

    if hasattr(sim_matrix, 'index'):
        final_partition = final_partition.set_index(sim_matrix.index)

    if len(n_clusters) == 1 and not hasattr(sim_matrix, 'index'):
        final_partition = final_partition.values.reshape(-1)

    return final_partition


def get_normalized_sim_matrix(sim_matrix):
    sim_matrix_min = sim_matrix.min()
    sim_matrix_max = sim_matrix.max()

    if len(sim_matrix.shape) == 2:
        sim_matrix_min = sim_matrix_min.min()
        sim_matrix_max = sim_matrix_max.max()

    return ((sim_matrix - sim_matrix_min) / (sim_matrix_max - sim_matrix_min))


def get_sim_matrix_by_partition(sim_matrix, partition):
    # pick first partition
    part = partition.iloc[:, 0]

    sorted_index = []

    new_matrix_vals = np.empty(sim_matrix.shape)
    new_matrix_vals[:] = np.nan
    pval_matrix_sorted = pd.DataFrame(
        new_matrix_vals,
        index=sim_matrix.index.tolist(),
        columns=sim_matrix.index.tolist()
    )

    for k in np.unique(part):
        cluster_index = part[part == k].index.tolist()
        sorted_index.extend(cluster_index)

        pval_matrix_sorted.loc[cluster_index, cluster_index] = sim_matrix.loc[cluster_index, cluster_index]

    return pval_matrix_sorted.loc[sorted_index, sorted_index]


def get_pval_matrix_by_partition(data, partition, k_internal, min_n_common_features, n_perms, n_jobs):
    # pick first partition
    part = partition.iloc[:, 0]

    sorted_index = []

    new_matrix_vals = np.empty((data.shape[0], data.shape[0]))
    new_matrix_vals[:] = np.nan
    pval_matrix_sorted = pd.DataFrame(
        new_matrix_vals,
        index=data.index.tolist(),
        columns=data.index.tolist()
    )

    for k in np.unique(part):
        cluster_index = part[part == k].index.tolist()

        cluster_k_pvalue_matrix = \
            calculate_simmatrix(data.loc[cluster_index], internal_n_clusters=k_internal, compute_perm_pvalue=True,
                                n_perm=n_perms, return_pvalue=True, min_n_common_features=min_n_common_features,
                                n_jobs=n_jobs)[1]

        sorted_index.extend(cluster_index)

        pval_matrix_sorted.loc[cluster_index, cluster_index] = cluster_k_pvalue_matrix.loc[cluster_index, cluster_index]

    return pval_matrix_sorted.loc[sorted_index, sorted_index]


def get_partition_spectral(sim_matrix, n_clusters, **kwargs):
    if len(sim_matrix.shape) == 1:
        sim_matrix = get_squareform(sim_matrix)

    if not hasattr(n_clusters, '__iter__'):
        n_clusters = (n_clusters,)

    norm_sim_matrix = get_normalized_sim_matrix(sim_matrix)

    def clustering_method(k):
        return SpectralClustering(n_clusters=k, affinity='precomputed', **kwargs).fit_predict(norm_sim_matrix)

    return _get_partition(clustering_method, n_clusters, sim_matrix)


def get_partition_agglomerative(sim_matrix, n_clusters, linkage_method='average', criterion='maxclust'):
    if len(sim_matrix.shape) == 1:
        sim_matrix = get_squareform(sim_matrix)

    if not hasattr(n_clusters, '__iter__'):
        n_clusters = (n_clusters,)

    # condensed disimilarity matrix for hierarchical methods
    norm_sim_matrix = get_normalized_sim_matrix(sim_matrix)
    norm_disim_pdist = 1 - squareform(norm_sim_matrix, checks=False)

    # FIXME: maybe we should warn if method is centroid, median or ward, as distances are not euclidean
    linkage_matrix = linkage(norm_disim_pdist, method=linkage_method, metric=None)

    def clustering_method(k):
        return fcluster(linkage_matrix, k, criterion=criterion)

    return _get_partition(clustering_method, n_clusters, sim_matrix), linkage_matrix


def clustermatch(data, n_clusters, **kargs):
    sim_matrix = calculate_simmatrix(data, **kargs)

    return get_partition_spectral(sim_matrix, n_clusters), get_normalized_sim_matrix(sim_matrix)


def run_kmeans(data, k, **kwargs):
    kmeans_n_init = 3
    if 'kmeans_n_init' in kwargs:
        kmeans_n_init = kwargs['kmeans_n_init']

    kmeans_random_state = None
    if 'kmeans_random_state' in kwargs:
        kmeans_random_state = kwargs['kmeans_random_state']

    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=kmeans_n_init, random_state=kmeans_random_state)

    return kmeans.fit_predict(data.reshape(-1, 1))
