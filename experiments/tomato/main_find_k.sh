#!/usr/bin/env bash
set -e

TOP_K=10
N_REPS=100
N_JOBS=2

export PYTHONPATH=~/projects/clustermatch_public/

python main_find_k.py \
    --harvest 0809 \
    --top-k ${TOP_K} \
    --n-runs-per-k ${N_REPS} \
    --n-jobs ${N_JOBS}

python main_find_k.py \
    --harvest 0910 \
    --top-k ${TOP_K} \
    --n-runs-per-k ${N_REPS} \
    --n-jobs ${N_JOBS}

python main_find_k.py \
    --harvest 1112 \
    --top-k ${TOP_K} \
    --n-runs-per-k ${N_REPS} \
    --n-jobs ${N_JOBS}

python main_find_k.py \
    --harvest all \
    --top-k ${TOP_K} \
    --n-runs-per-k ${N_REPS} \
    --n-jobs ${N_JOBS}
