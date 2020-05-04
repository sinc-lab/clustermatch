import numpy as np
from minepy.mine import MINE
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import SpectralClustering, AffinityPropagation, DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.optics import optics

from clustermatch.cluster import cm, calculate_simmatrix, get_partition_spectral
from clustermatch.utils.methods import distcorr


SPECTRAL_METHOD = 'spectral'
PAM_METHOD = 'pam'
AFFINITY_PROPAGATION_METHOD = 'aff-prop'
DBSCAN_METHOD = 'dbscan'
OPTICS_METHOD = 'optics'
HC_PREFIX_METHODS = 'hc-'


def get_part_from_clusters(clusters, n_objects):
    part = np.full(n_objects, np.nan)

    for clus_idx, clus in enumerate(clusters):
        part[clus] = clus_idx

    return part


def _get_partition_score(distance_matrix, part, medoids):
    distance_sum = 0

    for cluster_idx in range(len(part)):
        cluster_medoid = medoids[cluster_idx]
        cluster_points_idx = part[cluster_idx]
        cluster_points_dist_to_medoid = distance_matrix[cluster_points_idx, cluster_medoid]

        distance_sum += np.sum(cluster_points_dist_to_medoid)

    return distance_sum


def _run_pam(distance_matrix, k, n_reps=1):
    n_objects = distance_matrix.shape[0]

    parts = []
    final_parts = []
    parts_medoids = []
    parts_score = []

    for rep_idx in range(n_reps):
        initial_medoids = np.random.choice(n_objects, k)
        kmedoids_instance = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
        kmedoids_instance.process()

        part = kmedoids_instance.get_clusters()
        parts.append(part)

        final_part = get_part_from_clusters(part, n_objects)
        final_parts.append(final_part)

        part_medoids = kmedoids_instance.get_medoids()
        parts_medoids.append(part_medoids)

        part_score = _get_partition_score(distance_matrix, part, part_medoids)
        parts_score.append(part_score)

    best_part_idx = np.argmin(parts_score)
    return final_parts[best_part_idx]


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

    elif clustering_algorithm.startswith(HC_PREFIX_METHODS) or clustering_algorithm in (DBSCAN_METHOD, PAM_METHOD, OPTICS_METHOD):
        dist_data_matrix = 1 - sim_data_matrix
        n_objects = dist_data_matrix.shape[0]

        assert np.abs((dist_data_matrix - dist_data_matrix.T)).max() < 1e12

        if clustering_algorithm.startswith(HC_PREFIX_METHODS):
            # convert matrix to condensed form for hc
            dist_data_matrix = squareform(dist_data_matrix, checks=False)

            hc_method = clustering_algorithm.split('-')[1]
            z = linkage(dist_data_matrix, method=hc_method, metric=None)
            return fcluster(z, k, criterion='maxclust')

        elif clustering_algorithm == DBSCAN_METHOD:
            return DBSCAN(metric='precomputed', n_jobs=n_jobs).fit_predict(dist_data_matrix)

        elif clustering_algorithm == PAM_METHOD:
            return _run_pam(dist_data_matrix, k, n_reps=10)

        elif clustering_algorithm == OPTICS_METHOD:
            radius = 2.0
            neighbors = 3
            optics_instance = optics(dist_data_matrix, radius, neighbors, k, data_type='distance_matrix')

            optics_instance.process()

            return get_part_from_clusters(optics_instance.get_clusters(), n_objects)

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
