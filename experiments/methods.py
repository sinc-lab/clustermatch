import numpy as np
from minepy.mine import MINE
from scipy.cluster.hierarchy import average, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances

from clustermatch.cluster import calculate_simmatrix, get_partition_agglomerative, get_partition_spectral
from utils.methods import distcorr


def _mic(x, y):
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(x, y)
    return mine.mic()


def _compute_pearson(x, y):
    return abs(pearsonr(x, y)[0])


def run_spectral_pearson(data, k, n_jobs=1):
    """
    01. SC-Pearson
    """
    r_sim_mat = pairwise_distances(data, metric=_compute_pearson, n_jobs=n_jobs)
    # r_sim_mat = squareform(r_sim_vect)
    np.fill_diagonal(r_sim_mat, 1.0)
    norm_r_sim_mat = ((r_sim_mat - r_sim_mat.min()) / (r_sim_mat.max() - r_sim_mat.min()))
    return SpectralClustering(n_clusters=k,
                              affinity='precomputed')\
        .fit_predict(norm_r_sim_mat)


def _compute_spearman(x, y):
    return abs(spearmanr(x, y)[0])


def run_spectral_spearman(data, k, n_jobs=1):
    """
    02. SC-Spearman
    """
    r_sim_mat = pairwise_distances(data, metric=_compute_spearman, n_jobs=n_jobs)
    # r_sim_vect = pdist(data, lambda x, y: abs(spearmanr(x, y)[0]))
    # r_sim_mat = squareform(r_sim_vect)
    np.fill_diagonal(r_sim_mat, 1.0)
    norm_r_sim_mat = ((r_sim_mat - r_sim_mat.min()) / (r_sim_mat.max() - r_sim_mat.min()))
    return SpectralClustering(n_clusters=k, affinity='precomputed')\
        .fit_predict(norm_r_sim_mat)


def _compute_distcorr(x, y):
    return abs(distcorr(x, y))


def run_spectral_distcorr(data, k, n_jobs=1):
    """
    03. SC-DC
    """
    r_sim_mat = pairwise_distances(data, metric=_compute_distcorr, n_jobs=n_jobs)
    # r_sim_vect = pdist(data, lambda x, y: abs(distcorr(x, y)))
    # r_sim_mat = squareform(r_sim_vect)
    np.fill_diagonal(r_sim_mat, 1.0)
    norm_r_sim_mat = ((r_sim_mat - r_sim_mat.min()) / (r_sim_mat.max() - r_sim_mat.min()))
    return SpectralClustering(n_clusters=k, affinity='precomputed')\
        .fit_predict(norm_r_sim_mat)


def _compute_mic(x, y):
    return abs(_mic(x, y))


def run_spectral_mic(data, k, n_jobs=1):
    """
    04. SC-MIC
    """
    r_sim_mat = pairwise_distances(data, metric=_compute_mic, n_jobs=n_jobs)
    # r_sim_vect = pdist(data, lambda x, y: abs(_mic(x, y)))
    # r_sim_mat = squareform(r_sim_vect)
    np.fill_diagonal(r_sim_mat, 1.0)
    norm_r_sim_mat = ((r_sim_mat - r_sim_mat.min()) / (r_sim_mat.max() - r_sim_mat.min()))
    return SpectralClustering(n_clusters=k, affinity='precomputed')\
        .fit_predict(norm_r_sim_mat)


#########
# quantiles (internal clustering)
#########


def run_clustermatch_spectral_quantiles_k_medium(data, k, n_jobs=1):
    """
    00. Clustermatch
    """
    return _run_clustermatch_spectral_quantiles_generic(data, k, internal_n_clusters=range(2, 10 + 1), n_jobs=n_jobs)


def _run_clustermatch_spectral_quantiles_generic(data, k, **kwargs):
    sim_matrix = calculate_simmatrix(data, **kwargs)
    return get_partition_spectral(sim_matrix, n_clusters=k)
