import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler

clinc_sample_rate = 36931.8
'''filterOpts = {
    'high': {
        'Wn': 1000.,
        'N': 2,
        'btype': 'high',
        'ftype': 'butter'
    },
}'''

clinc_sample_interval = pd.Timedelta(27077, unit='ns').to_timedelta64()
clinc_sample_interval_sec = float(clinc_sample_interval) * 1e-9
clinc_sample_rate = (clinc_sample_interval_sec) ** -1

folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/20231109-Phoenix")
# file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555"]
file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555"]

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name_list = [
    'MB_1700670158_174163', 'MB_1700671071_947699', 'MB_1700671568_714180',
    'MB_1700672329_741498', 'MB_1700672668_26337', 'MB_1700673350_780580'
    ]

file_name_list = ['MB_1700673350_780580']
for file_name in file_name_list:
    clinc_df = pd.read_parquet(folder_path / (file_name + '_clinc.parquet'))
    artifact_df = pd.read_parquet(folder_path / (file_name + '_average_zscore.parquet'))
    reref_df = pd.read_parquet(folder_path / (file_name + '_clinc_reref.parquet'))
    # filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), clinc_sample_rate)

    stim_info = pd.read_parquet(folder_path / (file_name + '_stim_info.parquet'))

    left_sweep = 0
    right_sweep = 9e-3
    samples_left = int(-left_sweep / clinc_sample_interval_sec)
    samples_right = int(right_sweep / clinc_sample_interval_sec)
    t = np.arange(-samples_left, samples_right) * clinc_sample_interval_sec
    num_samples = t.shape[0]

    epoched_dict = {}
    epoched_artifact_dict = {}
    reref_dict = {}
    epoch_labels = ['timestamp', 'eid', 'amp', 'freq', 'pw']
    for timestamp, group in stim_info.groupby(['timestamp']):
        key = tuple(group.reset_index().loc[0, epoch_labels])
        first_index = np.flatnonzero(clinc_df.index > timestamp)[0]

        epoched_dict[key] = clinc_df.iloc[first_index:first_index + num_samples, :].copy()
        epoched_dict[key].index = t
        epoched_dict[key] = epoched_dict[key] - epoched_dict[key].mean()

        epoched_artifact_dict[key] = artifact_df.iloc[first_index:first_index + num_samples, :].copy()
        epoched_artifact_dict[key].index = t

        reref_dict[key] = reref_df.iloc[first_index:first_index + num_samples, :].copy()
        reref_dict[key].index = t
        reref_dict[key] = reref_dict[key] - reref_dict[key].mean()

    lfp_df = pd.concat(epoched_dict, names=epoch_labels + ['t'])
    lfp_df.columns.name = 'channel'
    lfp_df.to_parquet(folder_path / (file_name + '_epoched_lfp.parquet'))

    reref_lfp_df = pd.concat(reref_dict, names=epoch_labels + ['t'])
    reref_lfp_df.columns.name = 'channel'
    reref_lfp_df.to_parquet(folder_path / (file_name + '_epoched_reref_lfp.parquet'))

    epoched_artifact_df = pd.concat(epoched_artifact_dict, names=epoch_labels + ['t'])
    epoched_artifact_df.columns.name = 'channel'
    epoched_artifact_df.to_parquet(folder_path / (file_name + '_epoched_artifact.parquet'))


