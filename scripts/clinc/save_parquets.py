from isicpy.third_party.pymatreader import hdf5todict
from isicpy.utils import makeFilterCoeffsSOS
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import signal
import json, yaml, os

clinc_sample_rate = 36931.8
sid_to_intan = {
    1: 8,
    22: 12,
    18: 9,
    19: 7,
    23: 4,
    16: 6,
    15: 10,
    12: 27,
    11: 17,
    6: 21,
    14: 25,
    7: 5,
    0: 13
}

sid_to_label = {
    1: 'S1_S3',
    22: 'S22',
    18: 'S18',
    19: 'S19',
    23: 'S23',
    16: 'S16',
    15: 'S15',
    12: 'S12_S20',
    11: 'S11',
    6: 'S6',
    14: 'S14',
    7: 'S7',
    0: 'S0_S2'
}

# folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/20231109-Phoenix")
# file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555", 'MB_1699560792_657674']

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name_list = [
    'MB_1700670158_174163', 'MB_1700671071_947699', 'MB_1700671568_714180',
    'MB_1700672329_741498', 'MB_1700672668_26337', 'MB_1700673350_780580'
    ]
file_name_list = [
    'MB_1700672329_741498', 'MB_1700672668_26337', 'MB_1700673350_780580'
    ]

for file_name in file_name_list:
    print(file_name)
    file_path = folder_path / (file_name + '_f.mat')
    file_timestamp_parts = file_name.split('_')
    file_start_time = pd.Timestamp(float('.'.join(file_timestamp_parts[1:3])), unit='s', tz='EST')

    if os.path.exists(folder_path / 'yaml_lookup.json'):
        with open(folder_path / 'yaml_lookup.json', 'r') as f:
            yml_path = json.load(f)[file_name]
        with open(folder_path / yml_path, 'r') as f:
            routing_info_str = ''.join(f.readlines())
            routing_info = yaml.safe_load(routing_info_str.replace('\t', '  '))
        sid_to_eid = {c['sid']: c['eid'] for c in routing_info['data']['contacts']}
        clinc_col_names = [f'E{sid_to_eid[sid]}' for sid, _ in sid_to_intan.items() if sid in sid_to_eid]
        clinc_indexes = [value for sid, value in sid_to_intan.items() if sid in sid_to_eid]
        what_are_cols = 'eid'
    else:
        clinc_col_names = [f'S{sid}' for sid, value in sid_to_intan.items()]
        clinc_indexes = [value for sid, value in sid_to_intan.items()]
        what_are_cols = 'sid'

    with h5py.File(file_path, 'r') as hdf5_file:
        data = hdf5todict(hdf5_file, variable_names=['data_this_file'], ignore_fields=None)
    print('\tLoaded mat file.')

    clinc_sample_counts = data['data_this_file']['SampleCount'].astype(int)

    clinc_sample_counts = clinc_sample_counts - clinc_sample_counts[0]
    n_samples_total = clinc_sample_counts[-1] + 1

    clinc_index = pd.Index(range(n_samples_total))
    clinc_t_index = pd.timedelta_range(
        start=0, periods=n_samples_total,
        freq=pd.Timedelta(clinc_sample_rate ** -1, unit='s'),
        )

    valid_data_mask = pd.Series(clinc_index).isin(clinc_sample_counts).to_numpy()
    clinc_data_blank = np.zeros((n_samples_total, len(clinc_col_names)))
    clinc_data_blank[valid_data_mask, :] = data['data_this_file']['ChannelData'][:, clinc_indexes] * 0.195
    #  0.195 uV/count
    clinc_data = pd.DataFrame(clinc_data_blank, index=clinc_t_index, columns=clinc_col_names)
    clinc_data.columns.name = what_are_cols
    del clinc_data_blank

    clinc_trigs_blank = np.zeros((n_samples_total, 1))
    clinc_trigs_blank[valid_data_mask, :] = data['data_this_file']['SyncWave'].reshape(-1, 1)
    clinc_trigs = pd.DataFrame(
        clinc_trigs_blank, index=clinc_t_index, columns=['sync_wave'])
    del clinc_trigs_blank

    valid_data_df = pd.DataFrame(
        valid_data_mask.reshape(-1, 1), index=clinc_t_index, columns=['valid_data']
        )
    clinc_data.to_parquet(folder_path / (file_name + '_clinc.parquet'))
    clinc_trigs.to_parquet(folder_path / (file_name + '_clinc_trigs.parquet'))
    valid_data_df.to_parquet(folder_path / (file_name + '_valid_data.parquet'))
    print('Done')

# dsi_block_list = ['Block0001', 'Block0002', 'WholeSession']
# dsi_block_list = ['Block0001', 'Block0002', 'Block0003', 'Block0004', 'Block0005']
dsi_block_list = []

for dsi_block_name in dsi_block_list:
    print(dsi_block_name)
    dsi_df = pd.read_csv(folder_path / f"{dsi_block_name}.csv", header=12, index_col=0, low_memory=False)
    dsi_df = dsi_df.applymap(lambda x: 0 if x == '     x' else x)
    dsi_df.loc[:, :] = dsi_df.astype(float)
    dsi_df.index = pd.DatetimeIndex(dsi_df.index, tz='EST')

    filterOpts = {
        'high': {
            'Wn': 2.,
            'N': 8,
            'btype': 'high',
            'ftype': 'butter'
        }
    }

    emg_sample_rate = 500.
    dsi_trig_sample_rate = 1000.

    filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), emg_sample_rate)

    print('\tFiltering EMG...')
    emg_df = dsi_df.iloc[::2, :-2].copy()
    emg_df = pd.DataFrame(
        signal.sosfiltfilt(filterCoeffs, emg_df - emg_df.mean(), axis=0),
        index=emg_df.index, columns=emg_df.columns)

    emg_df.to_parquet(folder_path / f"{dsi_block_name}_emg.parquet")

    dsi_trigs = dsi_df.iloc[:, -2:].copy()
    dsi_trigs.to_parquet(folder_path / f"{dsi_block_name}_dsi_trigs.parquet")

    print('\tDone...')
