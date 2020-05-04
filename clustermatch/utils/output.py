import os
import time
from os.path import dirname, join
import shutil
import json

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform, pdist
import pandas as pd

try:
    import matplotlib
    import seaborn as sns
    sns.set(context="paper", font="monospace")
    MATPLOTLIB_INSTALLED = True
except:
    MATPLOTLIB_INSTALLED = False

try:
    import requests
    REQUESTS_INSTALLED = True
except:
    REQUESTS_INSTALLED = False

from clustermatch.utils.misc import get_temp_file_name


RESULTS_DIR = 'results'


def _get_condensed_distance_matrix(ensemble):
    return pdist(ensemble.T, lambda u, v: (u != v).sum() / len(u))


def get_timestamp():
    return time.strftime('%Y%m%d_%H%M%S')


def setup_results_dir(func):
    def func_wrapper(*args, **kwargs):
        results_dir = os.path.join(RESULTS_DIR, kwargs['timestamp'])
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        return func(*args, **kwargs)

    return func_wrapper


def get_clustergrammer_link(square_matrix, names):
    if not REQUESTS_INSTALLED:
        raise ValueError('requests is not installed')

    # save sim matrix as csv and get clustergrammer visualization
    df = pd.DataFrame(square_matrix)
    df['names'] = names
    df = df.set_index('names')
    df.columns = names
    square_matrix_file = get_temp_file_name('txt')
    df.to_csv(square_matrix_file, sep='\t', encoding='utf-8')

    clustergrammer_link = ''

    try:
        upload_url = 'http://amp.pharm.mssm.edu/clustergrammer/matrix_upload/'
        r = requests.post(upload_url, files={'file': open(square_matrix_file, 'rb')})
        clustergrammer_link = r.text
    except:
        pass

    return clustergrammer_link


@setup_results_dir
def to_binary(data, file_name, timestamp):
    file_path = os.path.join(RESULTS_DIR, timestamp, file_name + '.pkl')
    pd.to_pickle(data, file_path)


def from_binary(file_name):
    return pd.read_pickle(file_name)


@setup_results_dir
def create_partition_plot_html(partition, timestamp, sources=None):
    results_dir = os.path.join(RESULTS_DIR, timestamp)

    html_dir = join(dirname(__file__), 'html')

    for afile in os.listdir(html_dir):
        afile_path = join(html_dir, afile)
        shutil.copy(afile_path, results_dir)

    if sources is not None:
        sources = np.array(sources)
    else:
        sources = np.array([''] * len(partition))

    # FIXME: create json and copy
    json_path = os.path.join(RESULTS_DIR, timestamp, 'data' + '.json')

    cluster_children = []
    k_values = np.unique(partition)
    for cluster_number in k_values:
        idx = (partition == cluster_number).values
        cluster_objects = partition[idx].index.tolist()
        cluster_objects_sources = sources[idx]

        cluster_children.append(
            {'name': '',
             'children': [{'name': obj_name, 'source': obj_source, 'size': 1}
                          for obj_name, obj_source in zip(cluster_objects, cluster_objects_sources)]
             }
        )

    partition_json = {'name': '', 'children': cluster_children}

    with open(json_path, 'w') as jf:
        json.dump(partition_json, jf)

    return join(results_dir, 'index.html')


@setup_results_dir
def save_ensembles(ensembles, timestamp):
    for ens_name, ens in ensembles.items():
        ensemble_path = os.path.join(RESULTS_DIR, timestamp, ens_name + '_ensemble' + '.xls')
        ens.to_excel(ensemble_path, encoding='utf-8')


@setup_results_dir
def append_data_description(text, timestamp):
    filepath = os.path.join(RESULTS_DIR, timestamp, 'README.txt')
    with open(filepath, 'a') as f:
        f.write('\n')
        f.write(text + '\n')


@setup_results_dir
def write_text_file(text_content, file_name, timestamp):
    filepath = os.path.join(RESULTS_DIR, timestamp, file_name)
    with open(filepath, 'w') as f:
        f.write(text_content)


@setup_results_dir
def write_data_description(data_files, merged_sources, feature_names, sources_names, timestamp):
    filepath = os.path.join(RESULTS_DIR, timestamp, 'README.txt')
    with open(filepath, 'w') as f:
        f.write('Data files included:\n')
        for afile in data_files:
            f.write('  - {}\n'.format(afile))

        f.write('\n')
        f.write('Merged sources shape: {}\n'.format(merged_sources.shape))
        f.write('Number of features: {} ({} unique)\n'.format(len(feature_names), len(set(feature_names))))
        f.write('Number of sources: {}\n'.format(len(set(sources_names))))


@setup_results_dir
def save_excel(dataframe, filename, timestamp):
    filepath = os.path.join(RESULTS_DIR, timestamp, filename + '.xlsx')
    dataframe.to_excel(filepath, encoding='utf-8')


def save_partitions_simple(partitions, partitions_path, extra_columns=None, columns_order=None, sort_by_columns=None):
    if extra_columns is not None:
        extra_df = pd.DataFrame(extra_columns, index=partitions.index)
        partitions = pd.concat([partitions, extra_df], axis=1)

    if columns_order is not None:
        partitions = partitions[columns_order]

    if sort_by_columns is not None:
        partitions = partitions.sort_values(sort_by_columns)

    partitions.to_excel(partitions_path, encoding='utf-8')


@setup_results_dir
def save_partitions(partitions, timestamp, **kwargs):
    partitions_path = os.path.join(RESULTS_DIR, timestamp, 'partitions' + '.xls')
    save_partitions_simple(partitions, partitions_path, **kwargs)

    return partitions_path


@setup_results_dir
def save_coassociation_matrix(ensemble, partition, timestamp, columns_order=None, image_format='pdf'):
    if not MATPLOTLIB_INSTALLED:
        raise ValueError('matplotlib is not installed')

    condensed_matrix = _get_condensed_distance_matrix(ensemble)
    full_matrix = squareform(condensed_matrix)
    full_matrix = 1 - full_matrix
    np.fill_diagonal(full_matrix, 0.0)

    tomato_accessions = ensemble.columns

    if columns_order is None:
        ordering_partition = partition.columns[0]
    else:
        ordering_partition = columns_order[0]

    partition_idxs_sorted = np.argsort(partition[ordering_partition])
    sim_matrix_sorted = full_matrix[partition_idxs_sorted, :]
    sim_matrix_sorted = sim_matrix_sorted[:, partition_idxs_sorted]
    tomato_accessions_sorted = [tomato_accessions[idx] for idx in partition_idxs_sorted]

    df = pd.DataFrame(sim_matrix_sorted, index=tomato_accessions_sorted, columns=tomato_accessions_sorted)
    # fig = sns.plt.figure(figsize=(100, 100))
    ax = sns.heatmap(df, square=True, linewidths=.05, cmap='Blues', cbar=None)
    sns.plt.yticks(rotation=0)
    sns.plt.xticks(rotation=90)

    font_scale = (35.0 / len(tomato_accessions))
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(label.get_fontsize() * font_scale)

    sim_matrix_path = os.path.join(RESULTS_DIR, timestamp, 'coasociation_matrix.' + image_format)
    sns.plt.savefig(sim_matrix_path, dpi=300, bbox_inches='tight')
    sns.plt.close()

    sim_matrix_csv_path = os.path.join(RESULTS_DIR, timestamp, 'coasociation_matrix.csv')
    df.to_csv(sim_matrix_csv_path)

    return sim_matrix_path, full_matrix, sim_matrix_csv_path


@setup_results_dir
def save_similarity_matrix(partition, feature_names, sim_matrix, timestamp, image_format='pdf'):
    if not MATPLOTLIB_INSTALLED:
        raise ValueError('matplotlib is not installed')

    partition_idxs_sorted = np.argsort(partition)
    sim_matrix_sorted = sim_matrix[partition_idxs_sorted, :]
    sim_matrix_sorted = sim_matrix_sorted[:, partition_idxs_sorted]
    feature_names_sorted = [feature_names[idx] for idx in partition_idxs_sorted]

    df = pd.DataFrame(sim_matrix_sorted, index=feature_names_sorted, columns=feature_names_sorted)
    ax = sns.heatmap(df, square=True, linewidths=.05)

    sns.plt.yticks(rotation=0)
    sns.plt.xticks(rotation=90)

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(1.5)

    sim_matrix_path = os.path.join(RESULTS_DIR, timestamp, 'similarity_matrix.' + image_format)
    sns.plt.savefig(sim_matrix_path, dpi=300, bbox_inches='tight')

    return sim_matrix_path


@setup_results_dir
def save_reps_comparison(reps_comparison, timestamp):
    reps_comparison_csv = os.path.join(RESULTS_DIR, timestamp, 'reps_comparison.csv')
    reps_comparison.to_csv(reps_comparison_csv)
    return reps_comparison_csv


@setup_results_dir
def save_clustermap(sim_matrix, feature_names, sources_names, partition_linkage, timestamp, image_format='pdf'):
    if not MATPLOTLIB_INSTALLED:
        raise ValueError('matplotlib is not installed')

    font_scale = (50.0 / len(feature_names))
    # sns.set(font_scale=font_scale)

    fig = sns.plt.figure(figsize=(100, 100))

    df = pd.DataFrame(sim_matrix, index=feature_names, columns=feature_names)

    part00_k = 4
    part00 = fcluster(partition_linkage, part00_k, criterion='maxclust') - 1

    part01_k = 10
    part01 = fcluster(partition_linkage, part01_k, criterion='maxclust') - 1

    part02_k = 15
    part02 = fcluster(partition_linkage, part02_k, criterion='maxclust') - 1

    part03_k = 20
    part03 = fcluster(partition_linkage, part03_k, criterion='maxclust') - 1

    part00_colors = sns.color_palette('pastel', len(np.unique(part00)))
    part01_colors = sns.color_palette('pastel', len(np.unique(part01)))
    part02_colors = sns.color_palette('pastel', len(np.unique(part02)))
    part03_colors = sns.color_palette('pastel', len(np.unique(part03)))

    df_colors = pd.DataFrame(index=df.index)
    df_colors['$k$={0}'.format(part00_k)] = [part00_colors[i] for i in part00]
    df_colors['$k$={0}'.format(part01_k)] = [part01_colors[i] for i in part01]
    df_colors['$k$={0}'.format(part02_k)] = [part02_colors[i] for i in part02]
    df_colors['$k$={0}'.format(part03_k)] = [part03_colors[i] for i in part03]

    cm = sns.clustermap(df, col_linkage=partition_linkage, row_linkage=partition_linkage,
                        row_colors=df_colors, col_colors=df_colors)
    sns.plt.setp(cm.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    sns.plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

    ax = cm.cax
    pos1 = ax.get_position()  # get the original position
    pos2 = [pos1.x0 + 0.79, pos1.y0 - 0.02, pos1.width, pos1.height]
    ax.set_position(pos2)  # set a new position

    unique_sources = list(set(sources_names))
    sources_colors = sns.color_palette('hls', len(unique_sources))
    ylabels = [x for x in cm.ax_heatmap.get_yticklabels()]
    ylabels_text = [x.get_text() for x in ylabels]

    leg_fig = cm.ax_heatmap.get_yticklabels()[0].figure

    for label in cm.ax_heatmap.get_xticklabels():
        label_idx = feature_names.index(label.get_text())
        label_source = sources_names[label_idx]
        color_idx = unique_sources.index(label_source)
        label.set_color(sources_colors[color_idx])
        label.set_fontsize(label.get_fontsize() * font_scale)

        ylabel = ylabels[ylabels_text.index(label.get_text())]
        ylabel.set_color(sources_colors[color_idx])
        ylabel.set_fontsize(ylabel.get_fontsize() * font_scale)

    legend_labels = unique_sources
    legend_patches = [matplotlib.patches.Patch(color=C, label=L) for
                      C, L in zip(sources_colors, legend_labels)]
    leg_fig.legend(handles=legend_patches, labels=legend_labels, loc='lower right', bbox_to_anchor=(0.970, 0.05))

    clustermap_path = os.path.join(RESULTS_DIR, timestamp, 'clustermap.' + image_format)
    cm.savefig(clustermap_path, dpi=300, bbox_inches='tight')
    sns.plt.close()

    clustermap_csv_path = os.path.join(RESULTS_DIR, timestamp, 'clustermap.' + 'csv')
    df.to_csv(clustermap_csv_path)

    return clustermap_path, clustermap_csv_path

