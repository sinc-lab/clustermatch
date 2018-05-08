#!/usr/bin/env python
# coding=utf-8


import numpy as np

from experiments.data import blobs_data_generator02, transform_rows_nonlinear_and_categorical01
from experiments.execution import run_experiments_combination
from experiments.methods import run_clustermatch_spectral_quantiles_k_medium

# #################
# GLOBAL SETTINGS
#################

global_n_reps = 20


#################

np.random.seed(0)

###########################################
# Execution
###########################################

methods = (
    run_clustermatch_spectral_quantiles_k_medium,
)

data_generators = (blobs_data_generator02,)

# data_transformers = (
#     transform_rows_full_scaled01,
#     transform_rows_nonlinear01, transform_rows_nonlinear02, transform_rows_nonlinear03,
#     transform_rows_boxcox01, transform_rows_boxcox02, transform_rows_boxcox03,
#     transform_rows_boxcox04, transform_rows_boxcox05, transform_rows_boxcox06,
#     transform_rows_boxcox07,
# )

data_transformers = (
    transform_rows_nonlinear_and_categorical01,
)

data_noise_levels = (
    {'percentage_objects': 0.0, 'magnitude': 0.00, 'minmax_scale': True},
    {'percentage_objects': 1.0, 'magnitude': 0.10, 'minmax_scale': True},
    {'percentage_objects': 1.0, 'magnitude': 0.20, 'minmax_scale': True},
    {'percentage_objects': 1.0, 'magnitude': 0.30, 'minmax_scale': True},
    {'percentage_objects': 1.0, 'magnitude': 0.40, 'minmax_scale': True},
)

run_experiments_combination(
    global_n_reps,
    methods=methods,
    data_generators=data_generators,
    data_transformers=data_transformers,
    data_noise_levels=data_noise_levels,
)
