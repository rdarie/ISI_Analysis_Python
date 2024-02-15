import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from isicpy.clinc_lookup_tables import clinc_sample_rate
from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler

clinc_sample_interval_sec = float(clinc_sample_rate ** -1)

filterOptsClinc = {
    'low': {
        'Wn': 1500.,
        'N': 4,
        'btype': 'low',
        'ftype': 'butter'
    },
}
filterCoeffsClinc = makeFilterCoeffsSOS(filterOptsClinc.copy(), clinc_sample_rate)
apply_clinc_filters = False

folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/20231109-Phoenix")
file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555"]

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name_list = [
    'MB_1700670158_174163', 'MB_1700671071_947699',
    # 'MB_1700671568_714180',
    'MB_1700672329_741498', 'MB_1700672668_26337', 'MB_1700673350_780580'
    ]
file_name_list = ['MB_1700672668_26337', 'MB_1700673350_780580']
folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")
file_name_list = [
    "MB_1702047397_450767", "MB_1702048897_896568", "MB_1702049441_627410",
    "MB_1702049896_129326", "MB_1702050154_688487", "MB_1702051241_224335"
]
file_name_list = ["MB_1702050154_688487"]

# folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202401111300-Phoenix")
folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312211300-Phoenix")

routing_config_info = pd.read_json(folder_path / 'analysis_metadata/routing_config_info.json')
file_name_list = routing_config_info['child_file_name'].to_list()

per_pulse = False
has_reref = False

for file_name in file_name_list:
    clinc_df = pd.read_parquet(folder_path / (file_name + '_clinc.parquet'))
    if apply_clinc_filters:
        clinc_df = pd.DataFrame(
            signal.sosfiltfilt(filterCoeffsClinc, clinc_df, axis=0),
            index=clinc_df.index, columns=clinc_df.columns)
    artifact_df = pd.read_parquet(folder_path / (file_name + '_average_zscore.parquet'))
    if has_reref:
        reref_df = pd.read_parquet(folder_path / (file_name + '_clinc_reref.parquet'))
        if apply_clinc_filters:
            reref_df = pd.DataFrame(
                signal.sosfiltfilt(filterCoeffsClinc, reref_df, axis=0),
                index=reref_df.index, columns=reref_df.columns)
    if per_pulse:
        stim_info = pd.read_parquet(folder_path / (file_name + '_stim_info_per_pulse.parquet'))
    else:
        stim_info = pd.read_parquet(folder_path / (file_name + '_stim_info.parquet'))
    left_sweep = -2e-3
    right_sweep = 20e-3
    samples_left = int(left_sweep / clinc_sample_interval_sec)
    samples_right = int(right_sweep / clinc_sample_interval_sec)
    t = np.arange(samples_left, samples_right) * clinc_sample_interval_sec
    num_samples = t.shape[0]

    epoched_dict = {}
    epoched_artifact_dict = {}
    if has_reref:
        reref_dict = {}
    epoch_labels = ['timestamp', 'eid', 'amp', 'freq', 'pw', 'train_idx', 'rank_in_train']
    for timestamp, group in stim_info.groupby('timestamp'):
        key = tuple(group.reset_index().loc[0, epoch_labels])
        first_index = np.flatnonzero(clinc_df.index >= timestamp)[0]

        epoched_dict[key] = clinc_df.iloc[first_index + samples_left:first_index + samples_right, :].copy()
        epoched_dict[key].index = t
        epoched_dict[key] = epoched_dict[key]

        epoched_artifact_dict[key] = artifact_df.iloc[first_index + samples_left:first_index + samples_right, :].copy()
        epoched_artifact_dict[key].index = t

        if has_reref:
            reref_dict[key] = reref_df.iloc[first_index + samples_left:first_index + samples_right, :].copy()
            reref_dict[key].index = t
            reref_dict[key] = reref_dict[key]

    lfp_df = pd.concat(epoched_dict, names=epoch_labels + ['t'])
    lfp_df.columns.name = 'channel'
    file_name_suffix = '_per_pulse' if per_pulse else ''
    lfp_df.to_parquet(
        folder_path / (file_name + f'_epoched_lfp{file_name_suffix}.parquet'), engine='fastparquet')

    if has_reref:
        reref_lfp_df = pd.concat(reref_dict, names=epoch_labels + ['t'])
        reref_lfp_df.columns.name = 'channel'
        reref_lfp_df.to_parquet(
            folder_path / (file_name + f'_epoched_reref_lfp{file_name_suffix}.parquet'), engine='fastparquet')
    epoched_artifact_df = pd.concat(epoched_artifact_dict, names=epoch_labels + ['t'])
    epoched_artifact_df.columns.name = 'channel'
    epoched_artifact_df.to_parquet(
        folder_path / (file_name + f'_epoched_artifact{file_name_suffix}.parquet'), engine='fastparquet')
