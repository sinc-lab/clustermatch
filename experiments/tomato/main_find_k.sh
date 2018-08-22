#!/usr/bin/env bash

python main_find_k.py \
    --harvest 0809 \
    --top-k 10 \
    --n-runs-per-k 50 \
    --n-jobs 4

python main_find_k.py \
    --harvest 0910 \
    --top-k 10 \
    --n-runs-per-k 50 \
    --n-jobs 4

python main_find_k.py \
    --harvest 1112 \
    --top-k 10 \
    --n-runs-per-k 50 \
    --n-jobs 4

python main_find_k.py \
    --harvest all \
    --top-k 10 \
    --n-runs-per-k 50 \
    --n-jobs 4
