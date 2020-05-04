import os
import io
import argparse

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score as ari
from joblib import Parallel, delayed

from clustermatch.cluster import get_partition_spectral, calculate_simmatrix

from clustermatch.utils.data import merge_sources
from clustermatch.utils.output import to_binary, write_text_file


def sc(sim_matrix, k):
    return get_partition_spectral(sim_matrix, k).iloc[:, 0].values


def full_sample_and_combine(k, n_runs, sim_matrix):
    partitions = np.array([sc(sim_matrix, k) for i in range(n_runs)])
    aris = pdist(partitions, metric=ari)

    return {k: (aris.mean(), aris.std(), partitions)}


def run_experiment(sim_matrix, top_k, n_runs, n_jobs=1):
    results_by_k = Parallel(n_jobs=n_jobs)(
        delayed(full_sample_and_combine)(k, n_runs, sim_matrix)
        for k in range(2, top_k + 1)
    )

    results = {}
    for res in results_by_k:
        results.update(res)

    return results


BASE_DATA_DIR = 'data'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--harvest', required=True, choices=['all', '0809', '0910', '1112'])
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--n-runs-per-k', type=int, default=2)
    args = parser.parse_args()

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

    np.random.seed(33)

    timestamp = os.path.join(args.harvest, 'find_k')

    data_files = [os.path.join(BASE_DATA_DIR, df) for df in data_files_dict[args.harvest]]

    print('Merging data')
    merged_sources, feature_names, sources_names = merge_sources(data_files)

    print('Getting similarity matrix with Clustermatch')
    sim_matrix = calculate_simmatrix(merged_sources, n_jobs=args.n_jobs)
    to_binary(sim_matrix, 'sim_matrix', timestamp=timestamp)

    print('Running experiments')
    results = run_experiment(sim_matrix, args.top_k, args.n_runs_per_k, args.n_jobs)
    to_binary(results, 'results', timestamp=timestamp)

    print('#### Results ####')
    with io.StringIO() as results_content:
        results_content.write(f'Harvest: {args.harvest}\n')
        results_content.write(f'Top k: {args.top_k}\n')
        results_content.write(f'n runs per k: {args.n_runs_per_k}\n')
        results_content.write(f'n jobs: {args.n_jobs}\n')

        for k, res in results.items():
            msg = f'k={k}: {res[0]:.2f} ({res[1]:.2f})'
            print(msg)
            results_content.write(msg + '\n')

        write_text_file(results_content.getvalue(), f'k_results_{args.harvest}.txt', timestamp=timestamp)
