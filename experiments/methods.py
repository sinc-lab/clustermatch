import numpy as np
from minepy.mine import MINE
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import SpectralClustering, AffinityPropagation, DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

from clustermatch.cluster import cm, calculate_simmatrix, get_partition_spectral
from utils.methods import distcorr


SPECTRAL_METHOD= 'spectral'
AFFINITY_PROPAGATION_METHOD= 'aff-prop'
DBSCAN_METHOD='dbscan'
HC_PREFIX_METHODS= 'hc-'


def _run_clustering_generic(sim_data_matrix, k, clustering_algorithm, n_jobs=1):
    np.fill_diagonal(sim_data_matrix, 1.0)
    min_val, max_val = sim_data_matrix.min(), sim_data_matrix.max()
    sim_data_matrix = ((sim_data_matrix - min_val) / (max_val - min_val))

    if clustering_algorithm in (SPECTRAL_METHOD, AFFINITY_PROPAGATION_METHOD):
        if clustering_algorithm == SPECTRAL_METHOD:
            return SpectralClustering(n_clusters=k, affinity='precomputed', n_jobs=n_jobs).fit_predict(sim_data_matrix)

        elif clustering_algorithm == AFFINITY_PROPAGATION_METHOD:
            return AffinityPropagation(affinity='precomputed').fit_predict(sim_data_matrix)

        else:
            raise Exception('Programming error')

    elif clustering_algorithm.startswith(HC_PREFIX_METHODS) or clustering_algorithm in (DBSCAN_METHOD,):
        dist_data_matrix = 1 - sim_data_matrix
        # convert matrix to condensed form for hc
        dist_data_matrix = squareform(dist_data_matrix, checks=False)

        assert np.abs((dist_data_matrix - dist_data_matrix.T)).max() < 1e12

        if clustering_algorithm.startswith(HC_PREFIX_METHODS):
            hc_method = clustering_algorithm.split('-')[1]
            z = linkage(dist_data_matrix, method=hc_method, metric=None)
            return fcluster(z, k, criterion='maxclust')

        elif clustering_algorithm == DBSCAN_METHOD:
            dist_data_matrix = squareform(dist_data_matrix, checks=False)
            return DBSCAN(metric='precomputed', n_jobs=n_jobs).fit_predict(dist_data_matrix)

        else:
            raise Exception('Programming error')

    else:
        raise ValueError('Invalid clustering algorithm provided')


def _mic(x, y):
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(x, y)
    return mine.mic()


def _compute_pearson(x, y):
    return abs(pearsonr(x, y)[0])


def run_pearson(data, k, n_jobs=1, **kwargs):
    """
    01. SC-Pearson
    """
    r_sim_mat = pairwise_distances(data, metric=_compute_pearson, n_jobs=n_jobs)
    return _run_clustering_generic(r_sim_mat, k, n_jobs=n_jobs, **kwargs)


def _compute_spearman(x, y):
    return abs(spearmanr(x, y)[0])


def run_spearman(data, k, n_jobs=1, **kwargs):
    """
    02. SC-Spearman
    """
    r_sim_mat = pairwise_distances(data, metric=_compute_spearman, n_jobs=n_jobs)
    return _run_clustering_generic(r_sim_mat, k, n_jobs=n_jobs, **kwargs)


def _compute_distcorr(x, y):
    return abs(distcorr(x, y))


def run_distcorr(data, k, n_jobs=1, **kwargs):
    """
    03. SC-DC
    """
    r_sim_mat = pairwise_distances(data, metric=_compute_distcorr, n_jobs=n_jobs)
    return _run_clustering_generic(r_sim_mat, k, n_jobs=n_jobs, **kwargs)


def _compute_mic(x, y):
    return abs(_mic(x, y))


def run_mic(data, k, n_jobs=1, **kwargs):
    """
    04. SC-MIC
    """
    r_sim_mat = pairwise_distances(data, metric=_compute_mic, n_jobs=n_jobs)
    return _run_clustering_generic(r_sim_mat, k, n_jobs=n_jobs, **kwargs)


#########
# quantiles (internal clustering)
#########


def _compute_cm(x, y):
    return cm(x, y)[0]


def run_clustermatch_quantiles_k_medium(data, k, n_jobs=1, **kwargs):
    """
    00. Clustermatch
    """
    if data.dtype == object:
        # for categorical data use our internal functions
        r_sim_mat = calculate_simmatrix(data, **kwargs)
        return get_partition_spectral(r_sim_mat, n_clusters=k)
    else:
        # for numerical experiment, run it in the same way we run the rest of the methods
        r_sim_mat = pairwise_distances(data, metric=_compute_cm, n_jobs=n_jobs)
        return _run_clustering_generic(r_sim_mat, k, n_jobs=n_jobs, **kwargs)
