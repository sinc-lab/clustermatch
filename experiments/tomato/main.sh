#!/usr/bin/env bash
set -e

python main.py \
    --harvest 0809 \
    --spectral-n-init 50 \
    --compute-pvalues \
    --n-jobs 4

python main.py \
    --harvest 0910 \
    --spectral-n-init 50 \
    --compute-pvalues \
    --n-jobs 4

python main.py \
    --harvest 1112 \
    --spectral-n-init 50 \
    --compute-pvalues \
    --n-jobs 4

python main.py \
    --harvest all \
    --spectral-n-init 50 \
    --compute-pvalues \
    --n-jobs 4
