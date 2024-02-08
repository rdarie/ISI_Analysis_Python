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
dsi_block_list = []


folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/202311091300-Phoenix")
file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555", 'MB_1699560792_657674']
dsi_block_list = ['Block0001', 'Block0002']


folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name_list = [
    'MB_1700670158_174163', 'MB_1700671071_947699', 'MB_1700671568_714180',
    'MB_1700672329_741498', 'MB_1700672668_26337', 'MB_1700673350_780580'
    ]
dsi_block_list = ['Block0001', 'Block0002', 'Block0003', 'Block0004', 'Block0005']


folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202401221300-Benchtop")
file_name_list = [
    "MB_1702047397_450767",  "MB_1702048897_896568",  "MB_1702049441_627410",
    "MB_1702049896_129326",  "MB_1702050154_688487",  "MB_1702051241_224335"
    ]
file_name_list = []
# file_name_list = []
file_name_list = ['MB_1705952197_530018']

dsi_block_list = ['Block0001', 'Block0002', 'Block0003', 'Block0004', 'Block0005', 'Block0006']
dsi_block_list = ['Block0003', 'Block0004']
dsi_block_list = []
'''

# folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312191300-Phoenix")
folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202401091300-Phoenix")

with open(folder_path / 'analysis_metadata/general_metadata.json', 'r') as f:
    general_metadata = json.load(f)
    dsi_block_list = general_metadata['dsi_block_list']

# dsi_start_times = {}
for dsi_block_name in dsi_block_list:
    print(f"Loading {dsi_block_name}...")
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
