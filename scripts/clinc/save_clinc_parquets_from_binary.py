#!/usr/bin/env /users/rdarie/anaconda/isi_analysis/bin/python

import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
from matplotlib import pyplot as plt
from isicpy.clinc_lookup_tables import clinc_sample_rate, sid_to_intan
from pathlib import Path
import numpy as np
import pandas as pd
import json, yaml, os
from tqdm import tqdm
import gc
from ttictoc import tic, toc

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")

with open(folder_path / 'analysis_metadata/general_metadata.json', 'r') as f:
    general_metadata = json.load(f)
    file_name_list = general_metadata['file_name_list']

routing_configs_dict = {}
for file_name in file_name_list:
    print(f'Loading {file_name}...')
    tic()
    file_path = folder_path / (file_name + '_f.bin')
    file_timestamp_parts = file_name.split('_')
    file_start_time = pd.Timestamp(
        float('.'.join(file_timestamp_parts[1:3])), unit='s', tz='EST')

    record_dtype = np.dtype([
        ('ChannelData', 'int16', (64,)), ('SyncWave', 'int16', (1,)),
        ('StimStart', 'int16', (1,)), ('MuxConfig', 'int16', (1,)),
        ('SampleCount_MSB', 'uint16', (1,)), ('SampleCount_LSB', 'uint16', (1,)),
        ('USBPacketCount', 'int16', (1,))
    ])
    with open(file_path, 'rb') as _file:
        data = np.fromfile(_file, dtype=record_dtype)
    print(f'\tLoaded bin file ({data.shape[0]} samples ~ {data.shape[0] * 70 * 2 / 1e9:.2f} GB).')

    clinc_sample_counts = (
            data['SampleCount_MSB'].astype('int64') * 2 ** 16 +
            data['SampleCount_LSB'].astype('int64')).flatten()
    counts_series = pd.Series(clinc_sample_counts)

    ###  sanitize sample_counts
    bad_mask = (counts_series > counts_series.iloc[-1])
    bad_mask = bad_mask | (counts_series.diff().fillna(1.) <= 0)
    bad_mask = bad_mask | (counts_series.duplicated())
    good_mask = ~bad_mask.to_numpy()

    clinc_sample_counts = clinc_sample_counts[good_mask]
    data = data[good_mask]

    clinc_sample_counts = clinc_sample_counts - clinc_sample_counts[0]
    n_samples_total = clinc_sample_counts[-1] + 1
    # clinc_index = pd.Index(np.arange(n_samples_total))
    clinc_t_index = file_start_time + pd.TimedeltaIndex(np.arange(n_samples_total) / clinc_sample_rate, unit='s')

    clinc_data = pd.DataFrame(np.zeros((n_samples_total, data['ChannelData'].shape[1])), index=clinc_t_index)
    clinc_trigs = pd.DataFrame(np.zeros((n_samples_total, 1)), index=clinc_t_index, columns=['sync_wave'])

    valid_data_mask = pd.Series(np.arange(n_samples_total)).isin(clinc_sample_counts).to_numpy()
    valid_data_df = pd.DataFrame(
        valid_data_mask.reshape(-1, 1), index=clinc_t_index, columns=['valid_data']
        )

    memory_limited = True
    if memory_limited:
        print('\tAssembling dataframe...')
        chunk_size = int(5e5)
        index_lookup = pd.DataFrame(
            {
                'clinc_idx': clinc_sample_counts,
                'chunk': np.arange(clinc_sample_counts.shape[0]) // chunk_size
                },
            )
        for name, group in tqdm(index_lookup.groupby('chunk')):
            sample_num = group['clinc_idx'].to_numpy()
            data_index = group.index.to_numpy()
            clinc_data.iloc[sample_num, :] = data['ChannelData'][data_index].astype('float32') * 0.195  #  0.195 uV/count
            clinc_trigs.iloc[sample_num, :] = data['SyncWave'][data_index]
            gc.collect()
    else:
        clinc_data.iloc[clinc_sample_counts, :] = data['ChannelData'].astype('float32') * 0.195  #  0.195 uV/count
        clinc_trigs.iloc[clinc_sample_counts, :] = data['SyncWave']

    del data
    gc.collect()

    # get the HD64 routing
    # Try to get the HD64 routing from the log file
    log_info_csv = pd.read_csv(folder_path / f'{file_name}_log.csv')
    mask = log_info_csv['CODE'].isin(['config_hd64_e_switches'])
    if mask.any():
        config_info = log_info_csv.loc[mask, :].copy()
        config_info.loc[:, 'yml_path'] = config_info.apply(
            lambda x: folder_path / f'config_files/{Path(x["FILENAME"]).name}', axis='columns')
        config_info.rename(columns={'DATETIME': 'time', 'SEC_ELAPSED': 'sec_elapsed'}, inplace=True)
        config_info.drop(columns=['CODE', 'DETAILS', 'FILENAME'], inplace=True)
        config_info.loc[:, 'config_start_time'] = config_info['time'].apply(lambda x: pd.Timestamp(x, tz='EST'))
        end_times = config_info['config_start_time'].shift(-1)
        end_times.iloc[-1] = clinc_t_index[-1]
        config_info.loc[:, 'config_end_time'] = end_times
        clinc_col_names = []
        clinc_indexes = []
        for idx, row in config_info.iterrows():
            # print(f"\tLoading {row['yml_path']}")
            with open(row['yml_path'], 'r') as f:
                routing_info_str = ''.join(f.readlines())
                routing_info = yaml.safe_load(routing_info_str.replace('\t', '  '))
            sid_to_eid = {c['sid']: c['eid'] for c in routing_info['data']['contacts']}
            # print('\t\tsid_to_eid = ' + ' '.join([f'S{c["sid"]}:E{c["eid"]}' for c in routing_info['data']['contacts']]))
            present_sids = [sid for sid in sid_to_intan.keys() if sid in sid_to_eid.keys()]
            these_col_names = [f'E{sid_to_eid[sid]}' for sid, _ in sid_to_intan.items() if sid in sid_to_eid]
            # print('\t\tthese_col_names = ' + ' '.join(these_col_names))
            clinc_col_names.append(these_col_names)
            clinc_indexes.append([value for sid, value in sid_to_intan.items() if sid in sid_to_eid])
        config_info.loc[:, 'clinc_indexes'] = clinc_indexes
        config_info.loc[:, 'clinc_col_names'] = clinc_col_names
        what_are_cols = 'eid'
    else:
        ## TODO: handle manually written json files
        # if we couldn't find the yml path that way, maybe we wrote it down
        '''
        if (yml_path is None) and (os.path.exists(folder_path / 'analysis_metadata/yaml_lookup.json')):
            with open(folder_path / 'yaml_lookup.json', 'r') as f:
                yml_path = json.load(f)[file_name]
            with open(yml_path, 'r') as f:
                routing_info_str = ''.join(f.readlines())
                routing_info = yaml.safe_load(routing_info_str.replace('\t', '  '))
            sid_to_eid = {c['sid']: c['eid'] for c in routing_info['data']['contacts']}
            config_info = pd.Series(
                {
                    'config_start_time': clinc_t_index[0],
                    'config_end_time': clinc_t_index[-1],
                    'clinc_indexes': [f'E{sid_to_eid[sid]}' for sid, _ in sid_to_intan.items() if sid in sid_to_eid],
                    'clinc_col_names': [value for sid, value in sid_to_intan.items() if sid in sid_to_eid]

                }
            ).to_frame().T
            what_are_cols = 'eid'
        else:
            config_info = None
        '''
        config_info = None
    # if we found it by either method
    if config_info is None:
        config_info = pd.Series(
            {
                'config_start_time': clinc_t_index[0],
                'config_end_time': clinc_t_index[-1],
                'clinc_indexes': [value for sid, value in sid_to_intan.items()],
                'clinc_col_names': [f'S{sid}' for sid, value in sid_to_intan.items()]

            }
        ).to_frame().T
        what_are_cols = 'sid'

    print('\tSaving files...')
    config_info.loc[:, 'child_file_name'] = config_info.apply(
        lambda x: f"MB_{int(x['config_start_time'].timestamp())}",
        axis='columns')
    routing_configs_dict[file_name] = config_info.copy()
    for _, row in tqdm(config_info.iterrows()):
        time_mask = (clinc_t_index >= row['config_start_time']) & (clinc_t_index <= row['config_end_time'])
        data_subset = clinc_data.iloc[time_mask, row['clinc_indexes']].copy()
        data_subset.columns = row['clinc_col_names']
        data_subset.columns.name = what_are_cols
        trigs_subset = clinc_trigs.iloc[time_mask, :]
        validity_subset = valid_data_df.iloc[time_mask, :]
        #
        child_file_name = row['child_file_name']
        data_subset.to_parquet(folder_path / (child_file_name + '_clinc.parquet'), engine='fastparquet')
        trigs_subset.to_parquet(folder_path / (child_file_name + '_clinc_trigs.parquet'), engine='fastparquet')
        validity_subset.to_parquet(folder_path / (child_file_name + '_valid_clinc_data.parquet'), engine='fastparquet')
    print(f'\tDone. {toc() / 60:.2f} min. elapsed.')
all_configs = pd.concat(routing_configs_dict, names=['parent_file_name', 'log_idx']).reset_index()
all_configs.loc[:, 'yml_path'] = all_configs['yml_path'].apply(lambda x: f"{x}")
all_configs.to_json(folder_path / 'analysis_metadata/routing_config_info.json', indent=4)
print('{\n' + ',\n'.join([f'"{cn}": [""]' for cn in all_configs['child_file_name'].to_list()]) + '\n}')