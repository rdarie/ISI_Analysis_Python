import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
from matplotlib import pyplot as plt

from isicpy.third_party.pymatreader import hdf5todict
from isicpy.utils import makeFilterCoeffsSOS
from isicpy.clinc_lookup_tables import clinc_sample_rate, sid_to_intan, emg_sample_rate, dsi_trig_sample_rate
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import signal
import json, yaml, os


apply_emg_filters = True
if apply_emg_filters:
    filterOpts = {
        'high': {
            'Wn': .5,
            'N': 8,
            'btype': 'high',
            'ftype': 'butter'
        }
    }
    filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), emg_sample_rate)

'''
folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311071300-Phoenix")
file_name_list = [
    "MB_1699382682_316178", "MB_1699383052_618936", "MB_1699383757_778055", "MB_1699384177_953948",
    "MB_1699382925_691816", "MB_1699383217_58381", "MB_1699383957_177840"
    ]
dsi_block_list = []'''


folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/202311091300-Phoenix")
file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555", 'MB_1699560792_657674']
dsi_block_list = ['Block0001', 'Block0002']


'''folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name_list = [
    'MB_1700670158_174163', 'MB_1700671071_947699', 'MB_1700671568_714180',
    'MB_1700672329_741498', 'MB_1700672668_26337', 'MB_1700673350_780580'
    ]
dsi_block_list = ['Block0001', 'Block0002', 'Block0003', 'Block0004', 'Block0005']'''


folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")
file_name_list = [
    "MB_1702047397_450767",  "MB_1702048897_896568",  "MB_1702049441_627410",
    "MB_1702049896_129326",  "MB_1702050154_688487",  "MB_1702051241_224335"
]
dsi_block_list = ['Block0001', 'Block0002', 'Block0003', 'Block0004', 'Block0005', 'Block0006']
file_name_list = []
dsi_block_list = ['Block0003', 'Block0004']

for file_name in file_name_list:
    print(file_name)
    file_path = folder_path / (file_name + '_f.mat')
    file_timestamp_parts = file_name.split('_')
    file_start_time = pd.Timestamp(
        float('.'.join(file_timestamp_parts[1:3])), unit='s', tz='EST')

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

    clinc_sample_counts = pd.Series(data['data_this_file']['SampleCount'].astype(int))

    ###  sanitize sample_counts
    bad_mask = (clinc_sample_counts > clinc_sample_counts.iloc[-1])
    bad_mask = bad_mask | (clinc_sample_counts.diff().fillna(1.) <= 0)
    bad_mask = bad_mask | (clinc_sample_counts.duplicated())
    good_mask = ~bad_mask.to_numpy()

    data['data_this_file']['SampleCount'] = data['data_this_file']['SampleCount'][good_mask]
    clinc_sample_counts = data['data_this_file']['SampleCount'].astype(int)
    data['data_this_file']['ChannelData'] = data['data_this_file']['ChannelData'][good_mask, :]
    data['data_this_file']['SyncWave'] = data['data_this_file']['SyncWave'][good_mask]
    ###

    clinc_sample_counts = clinc_sample_counts - clinc_sample_counts[0]
    n_samples_total = clinc_sample_counts[-1] + 1

    clinc_index = pd.Index(range(n_samples_total))
    clinc_t_index = file_start_time + pd.TimedeltaIndex(np.arange(n_samples_total) / clinc_sample_rate, unit='s')

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
    clinc_data.to_parquet(folder_path / (file_name + '_clinc.parquet'), engine='fastparquet')
    clinc_trigs.to_parquet(folder_path / (file_name + '_clinc_trigs.parquet'), engine='fastparquet')
    valid_data_df.to_parquet(folder_path / (file_name + '_valid_clinc_data.parquet'), engine='fastparquet')
    print('Done')

# dsi_start_times = {}
for dsi_block_name in dsi_block_list:
    print(dsi_block_name)
    dsi_path = folder_path / f"{dsi_block_name}.csv"
    if not os.path.exists(dsi_path):
        dsi_path = folder_path / f"{dsi_block_name}.ascii"
    dsi_df = pd.read_csv(
        dsi_path,
        header=12, index_col=0, low_memory=False,
        parse_dates=True, infer_datetime_format=True)

    emg_cols = [cn for cn in dsi_df.columns if 'EMG' in cn]
    analog_cols = [cn for cn in dsi_df.columns if 'Input' in cn]

    dsi_start_time = pd.Timestamp(dsi_df.index[0], tz='EST')
    full_dsi_index = pd.timedelta_range(0, dsi_df.index[-1] - dsi_df.index[0], freq=pd.Timedelta(1, unit='ms'))
    full_emg_index = pd.timedelta_range(0, dsi_df.index[-1] - dsi_df.index[0], freq=pd.Timedelta(2, unit='ms'))

    dsi_df = dsi_df.applymap(lambda x: np.nan if x == '     x' else x)

    # raw_emg_df = dsi_df.loc[:, emg_cols].iloc[::2, :].copy().dropna().astype(float)
    raw_emg_df = dsi_df.loc[:, emg_cols].copy().dropna().astype(float)

    raw_emg_df.index = raw_emg_df.index - dsi_df.index[0]
    raw_trigs_df = dsi_df.loc[:, analog_cols].copy().dropna().astype(float)
    raw_trigs_df.index = raw_trigs_df.index - dsi_df.index[0]
    del dsi_df

    valid_trigs_mask = pd.Series(full_dsi_index).isin(raw_trigs_df.index).to_numpy()
    dsi_trigs_blank = np.zeros((full_dsi_index.shape[0], len(analog_cols)))
    dsi_trigs_blank[valid_trigs_mask, :] = raw_trigs_df.to_numpy()
    dsi_trigs = pd.DataFrame(
        dsi_trigs_blank,
        index=dsi_start_time + full_dsi_index, columns=analog_cols)
    dsi_trigs.columns.name = 'channel'
    del dsi_trigs_blank
    dsi_trigs.to_parquet(folder_path / f"{dsi_block_name}_dsi_trigs.parquet", engine='fastparquet')

    valid_data_df = pd.DataFrame(
        valid_trigs_mask.reshape(-1, 1), index=dsi_trigs.index, columns=['valid_data']
        )
    valid_data_df.to_parquet(folder_path / f"{dsi_block_name}_valid_dsi_trigs_mask.parquet", engine='fastparquet')

    valid_emg_mask = pd.Series(full_emg_index).isin(raw_emg_df.index).to_numpy()
    reverse_valid_emg_mask = pd.Series(raw_emg_df.index).isin(full_emg_index).to_numpy()

    emg_blank = np.zeros((full_emg_index.shape[0], len(emg_cols)))
    emg_blank[valid_emg_mask, :] = raw_emg_df.loc[reverse_valid_emg_mask, :].to_numpy()
    emg_df = pd.DataFrame(
        emg_blank, index=dsi_start_time + full_emg_index, columns=emg_cols)
    emg_df.columns.name = 'channel'
    del emg_blank

    if apply_emg_filters:
        print('\tFiltering EMG...')
        emg_df = pd.DataFrame(
            signal.sosfiltfilt(filterCoeffs, emg_df - emg_df.mean(), axis=0),
            index=emg_df.index, columns=emg_df.columns)
    else:
        emg_df = emg_df - emg_df.mean()

    emg_df.to_parquet(folder_path / f"{dsi_block_name}_emg.parquet", engine='fastparquet')

    valid_data_df = pd.DataFrame(
        valid_emg_mask.reshape(-1, 1), index=emg_df.index, columns=['valid_data']
        )
    valid_data_df.to_parquet(folder_path / f"{dsi_block_name}_valid_emg_mask.parquet", engine='fastparquet')
    print('\tDone...')

# output_times = pd.Series(dsi_start_times)
# output_times.to_json(folder_path / 'dsi_start_times.json')
