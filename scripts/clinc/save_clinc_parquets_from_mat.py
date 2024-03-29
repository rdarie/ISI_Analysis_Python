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

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312191300-Phoenix")
with open(folder_path / 'analysis_metadata/general_metadata.json', 'r') as f:
    general_metadata = json.load(f)
    file_name_list = general_metadata['file_name_list']

for file_name in file_name_list:
    print(file_name)
    file_path = folder_path / (file_name + '_f.mat')
    file_timestamp_parts = file_name.split('_')
    file_start_time = pd.Timestamp(
        float('.'.join(file_timestamp_parts[1:3])), unit='s', tz='EST')

    # get the HD64 routing
    yml_path = None
    # Try to get the HD64 routing from the log file
    log_info_csv = pd.read_csv(folder_path / f'{file_name}_log.csv')
    mask = log_info_csv['CODE'].isin(['config_hd64_e_switches'])
    relevant_codes = log_info_csv.loc[mask, :]
    if relevant_codes.shape[0] == 1:
        yml_filename = Path(relevant_codes.iloc[0, :]['FILENAME']).name
        yml_path = folder_path/ f'config_files/{yml_filename}'
    # if we couldn't find the yml path that way, maybe we wrote it down
    if (yml_path is None) and (os.path.exists(folder_path / 'analysis_metadata/yaml_lookup.json')):
        with open(folder_path / 'yaml_lookup.json', 'r') as f:
            yml_path = json.load(f)[file_name]
    # if we found it by either method
    if yml_path is not None:
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
