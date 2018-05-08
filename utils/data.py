import os
from collections import Counter
from zipfile import ZipFile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


def _process_sources(source_files):
    if not isinstance(source_files, (list, tuple)):
        source_files = [source_files]

    source_files_list = []
    source_files_list.extend(source_files)

    original_source_names_files = {}

    sources = {}

    while len(source_files_list) > 0:
        file_path = source_files_list.pop()

        file_name, file_ext = os.path.splitext(file_path)
        file_ext = file_ext.lower()
        file_name = os.path.basename(file_name)

        if file_ext == '.csv':
            sources[file_name] = pd.read_csv(file_path, index_col=0, mangle_dupe_cols=True)

        elif file_ext in ('.xls', '.xlsx'):
            excel_data = pd.read_excel(file_path, index_col=0, sheet_name=None)
            for sheet_name, sheet_data in excel_data.items():
                sheet_data.columns = sheet_data.columns.astype(str)

                # handle duplicated source names
                original_source_name = sheet_name
                source_name = original_source_name
                if source_name in sources.keys():
                    # rename previously added sources with original name
                    prev_source_data = sources.pop(source_name)
                    prev_source_file_name = original_source_names_files.pop(source_name)
                    prev_source_new_name = '{0} ({1})'.format(source_name, prev_source_file_name)
                    sources[prev_source_new_name] = prev_source_data

                    # new source name for current source data
                    source_name = '{0} ({1})'.format(source_name, file_name)

                sources[source_name] = sheet_data
                original_source_names_files[original_source_name] = file_name

        elif file_ext == '.zip':
            zipf = ZipFile(file_path)
            tempdir = TemporaryDirectory().name
            zipf.extractall(path=tempdir)

            for zip_member in zipf.infolist():
                source_files_list.append(os.path.join(tempdir, zip_member.filename))

    return sources


def get_sources(source_files, rep_merge=np.mean):
    key_func = lambda x: x.split('.')[0]

    sources = _process_sources(source_files)
    sources_names = []
    data_sources = []

    sources_names_sorted = sorted(sources.keys())

    for source_name in sources_names_sorted:
        source_data = sources[source_name]

        # # FIXME columns with '.' should not be allowed, however the code below breaks things
        # if np.array(['.' in col for col in source_data.columns]).sum():
        #     raise ValueError('Columns cannot have a point (.)')

        # FIXME we don't support aggregating values when there is one categorical variable and repeated columns
        numeric_rows = np.array([np.isreal(x) for x in source_data.iloc[:, 0]])

        processed_data = source_data.iloc[numeric_rows].groupby(key_func, axis=1).apply(rep_merge, axis=1)

        # FIXME related to previous comment.
        if processed_data.shape[1] != source_data.shape[1] and not numeric_rows.all():
            raise ValueError('Categorical rows with repeated columns is not supported.')

        if not numeric_rows.all():
            non_numeric_rows_names = source_data.index[~numeric_rows].tolist()
            processed_data = processed_data.append(source_data.loc[non_numeric_rows_names])

        # there can't be duplicated rows per each source
        if processed_data.index.duplicated().any():
            raise ValueError('There can\'t be duplicated rows per each source')

        data_sources.append(processed_data)
        sources_names.append(source_name)

    return data_sources, sources_names


def merge_sources(source_files, rep_merge=np.mean):
    processed_sources, sources_names = get_sources(source_files, rep_merge=rep_merge)
    sources_names = [sn for sn_idx, sn in enumerate(sources_names)
                     for i in range(processed_sources[sn_idx].shape[0])]

    full_sources = pd.concat(processed_sources)

    # renamed duplicated
    if not full_sources.index.is_unique:
        old_index = full_sources.index.tolist()
        index_counter = Counter(old_index)
        new_index = []

        for idx, index_value in enumerate(full_sources.index):
            index_value_count = index_counter[index_value]

            if index_value_count == 1:
                new_index.append(index_value)
            else:
                new_index_value = index_value + u' ({0})'.format(sources_names[idx])
                new_index.append(new_index_value)

        full_sources.index = pd.Index(new_index)

    full_sources.index.rename('features', inplace=True)

    return full_sources, list(full_sources.index), sources_names
