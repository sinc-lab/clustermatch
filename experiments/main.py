#!/usr/bin/env python

import numpy as np
import argparse

import experiments.data as exp_data
from experiments.data import blobs_data_generator02
from experiments.execution import run_experiments_combination
from experiments.methods import run_spectral_pearson, run_spectral_spearman, \
    run_clustermatch_spectral_quantiles_k_medium, run_spectral_distcorr, run_spectral_mic

parser = argparse.ArgumentParser()
parser.add_argument('data_transf', type=str)
parser.add_argument('noise_perc_obj', type=int)
parser.add_argument('n_jobs', type=int)
parser.add_argument('n_reps', type=int)
parser.add_argument('--n-features', default=100, type=int)
args = parser.parse_args()

# #################
# GLOBAL SETTINGS
#################

global_n_reps = args.n_reps


#################

np.random.seed(0)

###########################################
# Execution
###########################################

methods = (
    # run_agglo, run_kmeans,
    run_spectral_pearson,
    run_spectral_spearman,
    run_spectral_distcorr,
    run_spectral_mic,
    # run_clustermatch_spectral_kmeans_k_medium,
    run_clustermatch_spectral_quantiles_k_medium,
)

blob_gen = lambda: blobs_data_generator02(n_samples=args.n_features)
blob_gen.__doc__ = f"""
Blobs. n_features={args.n_features}, n_samples=1000, centers=3, cluster_std=0.10, center_box=(-1.0, 1.0)
"""
data_generators = (
    blob_gen,
)

data_transformers = (
    getattr(exp_data, args.data_transf),
)

data_noise_levels = (
    {'percentage_objects': args.noise_perc_obj / 100.0, 'percentage_measures': 0.00, 'magnitude': 0.00},
)

run_experiments_combination(
    global_n_reps,
    methods=methods,
    data_generators=data_generators,
    data_transformers=data_transformers,
    data_noise_levels=data_noise_levels,
    n_jobs=args.n_jobs,
)

