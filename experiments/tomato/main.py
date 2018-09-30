#!/usr/bin/env python
# coding=utf-8

import time
import os
import argparse

import numpy as np

from clustermatch.cluster import calculate_simmatrix, get_partition_spectral, get_normalized_sim_matrix, \
    get_pval_matrix_by_partition
from utils.data import merge_sources
from utils.output import get_timestamp, save_partitions, create_partition_plot_html, to_binary, get_clustergrammer_link, \
    save_excel, write_data_description, append_data_description


BASE_DATA_DIR = 'data'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--harvest', required=True, choices=['all', '0809', '0910', '1112'])
    parser.add_argument('--spectral-n-init', type=int, default=10)
    parser.add_argument('--n-clusters', type=int)
    parser.add_argument('--compute-pvalues', action='store_true')
    parser.add_argument('--n-jobs', type=int, default=1)
    args = parser.parse_args()

    np.random.seed(0)

    harvest_selected = args.harvest

    n_jobs = args.n_jobs

    # timestamp = get_timestamp()
    timestamp = harvest_selected

    # settings
    k_internal = None  # to run with several k values

    data_files_dict = {
        '0809': ['allSources08_09.xlsx'],
        '0910': ['allSources09_10.xlsx'],
        '1112': ['allSources11_12.xlsx'],
        'all': [
            'allSources08_09.xlsx',
            'allSources09_10.xlsx',
            'allSources11_12.xlsx',
        ],
    }

    data_files = [os.path.join(BASE_DATA_DIR, df) for df in data_files_dict[args.harvest]]

    print('Merging data')
    merged_sources, feature_names, sources_names = merge_sources(data_files)
    write_data_description(data_files, merged_sources, feature_names, sources_names, timestamp=timestamp)

    min_n_tomatoes_dict = {
        '0809': 6,
        '0910': 6,
        '1112': 6,
        'all': 6,
    }

    min_n_tomatoes = min_n_tomatoes_dict[harvest_selected]

    # k_final = (int(merged_sources.shape[0] * 0.40),)
    # k_final_dict = {
    #     '0809': 6,
    #     '0910': 5,
    #     '1112': 7,
    #     'all': 5,
    # }

    #k_final = (k_final_dict[harvest_selected], )
    k_final = (args.n_clusters, )

    columns_order = ['k={0}'.format(str(k)) for k in k_final]

    print('Getting similarity matrix with Clustermatch')

    if not args.compute_pvalues:
        compute_perm_pvalue = False
        n_perm = None
    else:
        compute_perm_pvalue = True
        n_perm = 500

    start_time = time.time()
    cm_sim_matrix, cm_pvalue_sim_matrix = \
        calculate_simmatrix(merged_sources, internal_n_clusters=k_internal, compute_perm_pvalue=compute_perm_pvalue,
                            n_perm=n_perm, return_pvalue=True, min_n_common_features=min_n_tomatoes,
                            n_jobs=n_jobs)
    print('Getting final partition')
    partition = get_partition_spectral(cm_sim_matrix, k_final, n_init=args.spectral_n_init, n_jobs=n_jobs)
    final_time = time.time()
    print('clustermatch, elapsed time: ' + str(final_time - start_time) + ' seconds.')

    if args.compute_pvalues:
        print('Getting pvalue matrix')
        cm_pvalue_sim_matrix = get_pval_matrix_by_partition(cm_pvalue_sim_matrix, partition)

        save_excel(cm_pvalue_sim_matrix, 'cm_pvalue', timestamp=timestamp)
        print('cm_pvalue saved')

    save_partitions(partition,
                    extra_columns={'sources': sources_names},
                    columns_order=['sources', *columns_order],
                    sort_by_columns=columns_order,
                    timestamp=timestamp)

    full_partition = partition[columns_order[0]]
    partition_plot_path = create_partition_plot_html(full_partition, timestamp=timestamp, sources=sources_names)

    print('Getting shared objects')
    shared_objects_matrix = calculate_simmatrix(merged_sources, sim_func='shared_objects', fill_diag_value=np.nan, n_jobs=n_jobs)
    shared_objects_matrix = get_pval_matrix_by_partition(shared_objects_matrix, partition)
    save_excel(shared_objects_matrix, 'shared_tomatoes', timestamp=timestamp)

