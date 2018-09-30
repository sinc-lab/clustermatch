#!/usr/bin/env bash
set -e

N_INIT=50
N_JOBS=2

export PYTHONPATH=~/projects/clustermatch_public/

python main.py \
    --harvest 0809 \
    --spectral-n-init ${N_INIT} \
    --n-clusters 4 \
    --n-jobs ${N_JOBS}

python main.py \
    --harvest 0910 \
    --spectral-n-init ${N_INIT} \
    --n-clusters 4 \
    --n-jobs ${N_JOBS}

python main.py \
    --harvest 1112 \
    --spectral-n-init ${N_INIT} \
    --n-clusters 7 \
    --n-jobs ${N_JOBS}

python main.py \
    --harvest all \
    --spectral-n-init ${N_INIT} \
    --n-clusters 7 \
    --n-jobs ${N_JOBS}

