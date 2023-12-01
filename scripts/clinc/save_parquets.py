import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.use('qtagg')   # generate interactive output
import matplotlib.pyplot as plt

from isicpy.third_party.pymatreader import hdf5todict
from isicpy.utils import makeFilterCoeffsSOS
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import signal

# folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/20231109-Phoenix")
# file_name_list = ["MB_1699558933_985097_f.mat", "MB_1699560317_650555_f.mat", 'MB_1699560792_657674_f.mat']

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name_list = [
    'MB_1700670158_174163_f.mat', 'MB_1700671071_947699_f.mat', 'MB_1700671568_714180_f.mat',
    'MB_1700672329_741498_f.mat', 'MB_1700672668_26337_f.mat', 'MB_1700673350_780580_f.mat'
    ]

file_name_list = []
for file_name in file_name_list:
    print(file_name)
    file_path = folder_path / file_name
    file_timestamp_parts = file_name.split('_')
    file_start_time = pd.Timestamp(float('.'.join(file_timestamp_parts[1:3])), unit='s', tz='EST')

    with h5py.File(file_path, 'r') as hdf5_file:
        data = hdf5todict(hdf5_file, variable_names=['data_this_file'], ignore_fields=None)

    clinc_slines = {
        'S1_S3': 8,
        'S22': 12,
        'S18': 9,
        'S19': 7,
        'S23': 4,
        'S16': 6,
        'S15': 10,
        'S12_S20': 27,
        'S11': 17,
        'S6': 21,
        'S14': 25,
        'S7': 5,
        'S0_S2': 13
        }

    clinc_col_names = [key for key, value in clinc_slines.items()]
    clinc_indexes = [value for key, value in clinc_slines.items()]

    clinc_sample_rate = 36931.8

    clinc_index = pd.timedelta_range(
        start=0, periods=data['data_this_file']['ChannelData'].shape[0],
        freq=pd.Timedelta(clinc_sample_rate ** -1, unit='s'),
        )
    # 'datetime64[us]'
    clinc_data = pd.DataFrame(
        data['data_this_file']['ChannelData'][:, clinc_indexes] * 0.195,  # 0.195 uV/count
        index=clinc_index, columns=clinc_col_names)

    print('\tSaving CLINC...')
    clinc_data.to_parquet(folder_path / file_name.replace('.mat', '_clinc.parquet'))

    clinc_trigs = pd.DataFrame(
        data['data_this_file']['SyncWave'].reshape(-1, 1),
        index=clinc_index, columns=['sync_wave']
    )

    clinc_trigs.to_parquet(folder_path / file_name.replace('.mat', '_clinc_trigs.parquet'))
    print('Done')

# dsi_block_list = ['Block0001', 'Block0002', 'WholeSession']
dsi_block_list = ['Block0001', 'Block0002', 'Block0003', 'Block0004', 'Block0005']
# dsi_block_list = []
for dsi_block_name in dsi_block_list:
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

    emg_df = dsi_df.iloc[::2, :-2].copy()
    emg_df = pd.DataFrame(
        signal.sosfiltfilt(filterCoeffs, emg_df - emg_df.mean(), axis=0),
        index=emg_df.index, columns=emg_df.columns)

    print('Saving EMG...')
    emg_df.to_parquet(folder_path / f"{dsi_block_name}_emg.parquet")

    dsi_trigs = dsi_df.iloc[:, -2:].copy()
    dsi_trigs.to_parquet(folder_path / f"{dsi_block_name}_dsi_trigs.parquet")
