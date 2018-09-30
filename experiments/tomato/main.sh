#!/usr/bin/env bash
set -e

N_INIT=50
N_JOBS=2
PVALUE_N_PERMS=10000

export PYTHONPATH=~/projects/clustermatch_public/

python main.py \
    --harvest 0809 \
    --spectral-n-init ${N_INIT} \
    --n-clusters 4 \
    --compute-pvalues \
    --compute-pvalues-n-perms ${PVALUE_N_PERMS} \
    --n-jobs ${N_JOBS}

python main.py \
    --harvest 0910 \
    --spectral-n-init ${N_INIT} \
    --n-clusters 4 \
    --compute-pvalues \
    --compute-pvalues-n-perms ${PVALUE_N_PERMS} \
    --n-jobs ${N_JOBS}

python main.py \
    --harvest 1112 \
    --spectral-n-init ${N_INIT} \
    --n-clusters 7 \
    --compute-pvalues \
    --compute-pvalues-n-perms ${PVALUE_N_PERMS} \
    --n-jobs ${N_JOBS}

python main.py \
    --harvest all \
    --spectral-n-init ${N_INIT} \
    --n-clusters 7 \
    --compute-pvalues \
    --compute-pvalues-n-perms ${PVALUE_N_PERMS} \
    --n-jobs ${N_JOBS}

