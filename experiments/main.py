#!/usr/bin/env python

import numpy as np
import argparse

import experiments.data as exp_data
from experiments.data import blobs_data_generator01, blobs_data_generator02, \
    transform_rows_nonlinear01, transform_rows_nonlinear02, transform_rows_nonlinear03, transform_rows_nonlinear04, \
    transform_rows_nonlinear05, transform_rows_nonlinear06, transform_rows_nonlinear07, transform_rows_nonlinear08, \
    transform_rows_nonlinear09, transform_rows_nonlinear10, transform_rows_nonlinear11, transform_rows_nonlinear12, \
    transform_rows_nonlinear03_01
from experiments.execution import run_experiments_combination
from experiments.methods import run_kmeans, run_agglo, run_spectral_pearson, run_spectral_spearman, \
    run_clustermatch_spectral_kmeans_k_medium, run_clustermatch_spectral_quantiles_k_medium, \
    run_spectral_distcorr, run_spectral_mic

parser = argparse.ArgumentParser()
parser.add_argument('data_transf', type=str)
parser.add_argument('noise_perc_obj', type=int)
parser.add_argument('n_jobs', type=int)
parser.add_argument('n_reps', type=int)
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
    run_agglo, run_kmeans,
    run_spectral_pearson,
    run_spectral_spearman,
    run_spectral_distcorr,
    run_spectral_mic,
    # run_clustermatch_spectral_kmeans_k_medium,
    run_clustermatch_spectral_quantiles_k_medium,
)

data_generators = (
    blobs_data_generator02,
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

