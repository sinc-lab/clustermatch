#!/usr/bin/env python

import numpy as np
import argparse

import experiments.data as exp_data
from experiments.data import blobs_data_generator02
from experiments.execution import run_experiments_combination
from experiments.methods import run_pearson, run_spearman, \
    run_clustermatch_quantiles_k_medium, run_distcorr, run_mic

parser = argparse.ArgumentParser()
parser.add_argument('--data-transf', required=True, type=str)
parser.add_argument('--noise-perc', required=True, type=int)
parser.add_argument('--k-final', type=int)
parser.add_argument('--clustering-algorithm', type=str, default='spectral',
                    choices=('spectral', 'hc-complete', 'hc-single', 'hc-average'))
parser.add_argument('--n-jobs', type=int, default=1)
parser.add_argument('--n-reps', type=int, default=1)
parser.add_argument('--n-features', default=100, type=int)
parser.add_argument('--clustering-metric', default='ari', type=str, choices=('ari', 'ami'))
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
    run_pearson,
    run_spearman,
    run_distcorr,
    run_mic,
    run_clustermatch_quantiles_k_medium,
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
    {'percentage_objects': args.noise_perc / 100.0, 'percentage_measures': 0.00, 'magnitude': 0.00},
)

run_experiments_combination(
    global_n_reps,
    methods=methods,
    data_generators=data_generators,
    data_transformers=data_transformers,
    data_noise_levels=data_noise_levels,
    n_jobs=args.n_jobs,
    clustering_algorithm=args.clustering_algorithm,
    metric=args.clustering_metric,
    k_final=args.k_final,
)
