import argparse
import logging

from clustermatch import __short_description__
from clustermatch.cluster import calculate_simmatrix, get_partition_spectral
from clustermatch.utils.data import merge_sources
from clustermatch.utils.output import save_partitions_simple

LOG_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger('root')


def run():
    parser = argparse.ArgumentParser(description=__short_description__)

    # Mandatory parameters
    parser.add_argument('-i', '--input-files', required=True, type=str, nargs='+', help=
        'Path to input data files (could be one or multiple files). It could be a csv, xls (with different worksheets) or zip file.'
    )
    parser.add_argument('-k', '--n-clusters', required=True, type=int, nargs='+', help=
        'Number of final clusters (could contain multiple values).'
    )
    parser.add_argument('-o', '--output-file', required=True, type=str, help=
        'Path to output data partition file. The extension'
    )

    # Optional parameters
    parser.add_argument('--n-init', type=int, default=10, help=
        'Number of time the k-means algorithm will be run with different centroid seeds. '
        'The final results will be the best output of n_init consecutive runs in terms of inertia.'
    )
    parser.add_argument('--n-jobs', type=int, default=1, help=
        'The number of parallel jobs to run. -1 means using all processors.'
    )
    parser.add_argument('--minimum-objects', type=int, default=5, help=
        'Minimum amount of objects shared between two features to process them.'
    )

    args = parser.parse_args()

    # Validate parameters
    if any(x < 2 for x in args.n_clusters):
        parser.error('Number of final clusters must be >= 2')

    # Read data files
    logger.info('Reading input data files')
    merged_sources, feature_names, sources_names = merge_sources(args.input_files)

    # Run clustermatch
    logger.info(f'Getting similarity matrix for {merged_sources.shape[0]} variables')
    cm_sim_matrix = calculate_simmatrix(merged_sources, min_n_common_features=args.minimum_objects, n_jobs=args.n_jobs)

    logger.info(f'Running spectral clustering with k={args.n_clusters}')
    partition = get_partition_spectral(cm_sim_matrix, args.n_clusters, n_init=args.n_init, n_jobs=args.n_jobs)

    # if args.compute_pvalues:
    #     print('Getting pvalue matrix')
    #     cm_pvalue_sim_matrix = get_pval_matrix_by_partition(
    #         merged_sources, partition,
    #         k_internal, min_n_tomatoes,
    #         args.compute_pvalues_n_perms,
    #         n_jobs
    #     )
    #
    #     save_excel(cm_pvalue_sim_matrix, 'cm_pvalue', timestamp=timestamp)
    #     print('cm_pvalue saved')

    columns_order = ['k={0}'.format(str(k)) for k in args.n_clusters]

    logger.info(f'Saving partition to {args.output_file}')
    save_partitions_simple(
        partition,
        args.output_file,
        extra_columns={'sources': sources_names},
        columns_order=['sources', *columns_order],
        sort_by_columns=columns_order
    )
