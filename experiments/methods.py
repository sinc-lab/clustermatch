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


def run_kmeans(data, k, **kwargs):
    """
    99. k-means (k-means++, euclidean)
    """
    return KMeans(n_clusters=k, n_init=1).fit_predict(data)


def run_agglo(data, k, **kwargs):
    """
    99. Agglomerative (average, euclidean)
    """
    sim_vect = pdist(data)
    z = average(sim_vect)
    return fcluster(z, k, criterion='maxclust')


def run_agglo_pearson(data, k):
    """
    01. Agglomerative (average, Pearson)
    """
    r_disim_vect = pdist(data, lambda x, y: 1 - abs(pearsonr(x, y)[0]))
    z = average(r_disim_vect)
    return fcluster(z, k, criterion='maxclust')


def run_agglo_spearman(data, k):
    """
    02. Agglomerative (average, Spearman)
    """
    r_disim_vect = pdist(data, lambda x, y: 1 - abs(spearmanr(x, y)[0]))
    z = average(r_disim_vect)
    return fcluster(z, k, criterion='maxclust')


def run_agglo_distcorr(data, k):
    """
    03. Agglomerative (average, Distance correlation)
    """
    r_disim_vect = pdist(data, lambda x, y: 1 - abs(distcorr(x, y)))
    z = average(r_disim_vect)
    return fcluster(z, k, criterion='maxclust')


def run_agglo_mic(data, k):
    """
    04. Agglomerative (average, MIC)
    """
    r_disim_vect = pdist(data, lambda x, y: 1 - abs(_mic(x, y)))
    z = average(r_disim_vect)
    return fcluster(z, k, criterion='maxclust')


def _compute_pearson(x, y):
    return abs(pearsonr(x, y)[0])


def run_spectral_pearson(data, k, n_jobs=1):
    """
    01. Spectral clustering (average, Pearson)
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
    02. Spectral clustering (average, Spearman)
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
    03. Spectral clustering (average, Distance Correlation)
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
    04. Spectral clustering (average, MIC)
    """
    r_sim_mat = pairwise_distances(data, metric=_compute_mic, n_jobs=n_jobs)
    # r_sim_vect = pdist(data, lambda x, y: abs(_mic(x, y)))
    # r_sim_mat = squareform(r_sim_vect)
    np.fill_diagonal(r_sim_mat, 1.0)
    norm_r_sim_mat = ((r_sim_mat - r_sim_mat.min()) / (r_sim_mat.max() - r_sim_mat.min()))
    return SpectralClustering(n_clusters=k, affinity='precomputed')\
        .fit_predict(norm_r_sim_mat)


#########
# k-means (internal clustering)
#########

def run_clustermatch_spectral_kmeans(data, k):
    """
    00. clustermatch (internal kmeans, Spectral clustering)
    """
    sim_matrix = calculate_simmatrix(data, internal_clustering_method='kmeans')
    return get_partition_spectral(sim_matrix, n_clusters=k)


def run_clustermatch_spectral_kmeans_k_medium(data, k):
    """
    00.05. clustermatch (Spectral, internal k medium, all partitions, max ARI)
    """
    return _run_clustermatch_spectral_kmeans_generic(data, k, internal_n_clusters=tuple(range(2, 10+1)))


def _run_clustermatch_spectral_kmeans_generic(data, k, internal_n_clusters=None):
    sim_matrix = calculate_simmatrix(data,
                                     internal_clustering_method='kmeans',
                                     internal_n_clusters=internal_n_clusters)
    return get_partition_spectral(sim_matrix, n_clusters=k)


#########
# quantiles (internal clustering)
#########

def run_clustermatch_spectral_quantiles(data, k, n_jobs=1):
    """
    00.10. clustermatch ari (Spectral, internal quantiles)
    """
    return _run_clustermatch_spectral_quantiles_generic(data, k, n_jobs=n_jobs)


def run_clustermatch_spectral_quantiles_k_medium(data, k, n_jobs=1):
    """
    00.20. clustermatch ari (Spectral, internal quantiles (k medium))
    """
    return _run_clustermatch_spectral_quantiles_generic(data, k, internal_n_clusters=range(2, 10 + 1), n_jobs=n_jobs)


def _run_clustermatch_spectral_quantiles_generic(data, k, **kwargs):
    sim_matrix = calculate_simmatrix(data, **kwargs)
    return get_partition_spectral(sim_matrix, n_clusters=k)
