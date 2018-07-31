import os
from json import JSONEncoder, dumps
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari, adjusted_mutual_info_score as ami
from tabulate import tabulate

from utils.output import get_timestamp


def _run_experiment(rep_number, data_generator, methods, k_final=None, data_transform=None, data_noise=None,
                    data_seed_mode=False, metric='ari', **kwargs):
    data, data_ref = data_generator(seed=(rep_number if data_seed_mode else None))

    if k_final is None:
        data_n_clusters = len(np.unique(data_ref))
    else:
        data_n_clusters = k_final

    if data_transform is not None:
        data = data_transform(data, data_noise=data_noise)

    for met in methods:
        met_name = str.strip(met.__doc__)

        start_time = time()

        part = met(data.copy(), data_n_clusters, **kwargs)

        end_time = time()
        met_time = end_time - start_time

        if metric == 'ari':
            part_metric_value = ari(data_ref, part)
        elif metric == 'ami':
            part_metric_value = ami(data_ref, part)

        yield (met_name, met_time, part_metric_value)


def _run_full_experiment(experiment_data, **kwargs):
    results = pd.DataFrame(columns=('data_transf', 'noise_perc_obj', 'noise_perc_mes', 'noise_mes_mag', 'rep', 'method', 'time', 'metric'))

    glox_idx = 0
    for i in range(experiment_data['n_reps']):
        for a_result in _run_experiment(i,
                                        data_generator=experiment_data['data_generator'],
                                        methods=experiment_data['methods'],
                                        data_transform=experiment_data['data_transform'] if 'data_transform' in experiment_data else None,
                                        data_noise=experiment_data['data_noise'] if 'data_noise' in experiment_data else None,
                                        **kwargs):

            data_transform_name = experiment_data['data_transform'].__name__

            results.loc[glox_idx] = (
                data_transform_name,
                experiment_data['data_noise']['percentage_objects'],
                experiment_data['data_noise']['percentage_measures'],
                experiment_data['data_noise']['magnitude'],
                i
            ) + a_result

            glox_idx += 1

    return results


class ExperimentEnconder(JSONEncoder):
    def default(self, o):
        if callable(o):
            return str.strip(o.__doc__)
        else:
            super(ExperimentEnconder, self).default(o)


def _get_experiment_description(experiment_data):
    experiment_data = {key:experiment_data[key] for key in experiment_data if key != 'methods'}
    return dumps(experiment_data, indent=2, cls=ExperimentEnconder, sort_keys=True)


def _get_summary_results(results, group_by='method', aggregate={'metric': ['mean', 'std'], 'time': 'mean'}):
    summary = results.groupby(group_by).agg(aggregate).round(2)
    return summary[sorted(list(aggregate.keys()))]


def run_experiments_combination(n_reps, data_generators, methods, data_transformers=(None,), data_noise_levels=(None,), **kwargs):
    results_dir = 'results_{}_{}'.format(data_transformers[0].__name__, data_noise_levels[0]['percentage_objects'])
    timestamp = get_timestamp()
    experiment_dir = os.path.join(results_dir, timestamp)
    os.makedirs(experiment_dir)

    final_results = pd.DataFrame(columns=('data_transf', 'noise_perc_obj', 'noise_perc_mes', 'noise_mes_mag', 'rep', 'method', 'time', 'metric'))

    glob_idx = 0
    for data_gen in data_generators:
        for data_trans in data_transformers:
            for data_noise in data_noise_levels:

                experiment_data = {
                    'n_reps': n_reps,
                    'methods': methods,
                    'data_generator': data_gen,
                    'data_transform': data_trans,
                    'data_noise': data_noise,
                    'clustering_metric': kwargs['metric'],
                    'clustering_algorithm': kwargs['clustering_algorithm'],
                    'k_final': kwargs['k_final'],
                }

                experiment_description = _get_experiment_description(experiment_data)
                print('Running now:\n' + experiment_description, flush=True)

                results = _run_full_experiment(experiment_data, **kwargs)
                final_results = final_results.append(results, ignore_index=True)
                pd.to_pickle(final_results, os.path.join(experiment_dir, 'final_results.pkl'))

                with open(os.path.join(experiment_dir, 'output{:03d}.txt'.format(glob_idx)), 'w') as txt:
                    txt.write(experiment_description + '\n\n' + tabulate(_get_summary_results(results), headers='keys', floatfmt='.2f'))

                glob_idx += 1
