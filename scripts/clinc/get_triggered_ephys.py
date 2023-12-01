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
for file_name in file_name_list:
    clinc_df = pd.read_parquet(folder_path / (file_name + '_f_clinc.parquet'))
    artifact_df = pd.read_parquet(folder_path / (file_name + '_average_zscore.parquet'))
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
    for timestamp, row in stim_info.iterrows():
        key = (timestamp, *row.to_list())
        first_index = np.flatnonzero(clinc_df.index > timestamp)[0]

        epoched_dict[key] = clinc_df.iloc[first_index:first_index + num_samples, :].copy()
        epoched_dict[key].index = t
        epoched_dict[key] = epoched_dict[key] - epoched_dict[key].mean()

        epoched_artifact_dict[key] = artifact_df.iloc[first_index:first_index + num_samples, :].copy()
        epoched_artifact_dict[key].index = t

    lfp_df = pd.concat(epoched_dict, names=['timestamp'] + stim_info.columns.to_list() + ['t'])
    lfp_df.columns.name = 'channel'
    lfp_df.to_parquet(folder_path / (file_name + '_epoched_lfp.parquet'))

    epoched_artifact_df = pd.concat(epoched_artifact_dict, names=['timestamp'] + stim_info.columns.to_list() + ['t'])
    epoched_artifact_df.columns.name = 'channel'
    epoched_artifact_df.to_parquet(folder_path / (file_name + '_epoched_artifact.parquet'))


