import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from isicpy.clinc_lookup_tables import clinc_sample_rate, emg_sample_rate, dsi_mb_clock_offsets
from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler, scale
import os

filterOpts = {
    'low': {
        'Wn': 50.,
        'N': 4,
        'btype': 'low',
        'ftype': 'butter'
    }
}
filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), emg_sample_rate)
scale_emg = True

emg_sample_interval_sec = float(emg_sample_rate ** -1)

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
routing_config_info = pd.read_json(folder_path / 'analysis_metadata/routing_config_info.json')
file_name_list = routing_config_info['child_file_name'].to_list()

per_pulse = False
for file_name in file_name_list:
    if per_pulse:
        stim_info_path = folder_path / (file_name + '_stim_info_per_pulse.parquet')
    else:
        stim_info_path = folder_path / (file_name + '_stim_info.parquet')
    if not os.path.exists(stim_info_path):
        continue
    with open(folder_path / 'analysis_metadata/dsi_block_lookup.json', 'r') as f:
        emg_block_name = json.load(f)[file_name][0]

    with open(folder_path / 'analysis_metadata/general_metadata.json', 'r') as f:
        clock_difference = json.load(f)["dsi_clock_difference"]
    with open(folder_path / 'analysis_metadata/dsi_to_mb_fine_offsets.json', 'r') as f:
        dsi_fine_offset = json.load(f)[file_name][emg_block_name]

    print(f'On {file_name}...')
    stim_info = pd.read_parquet(stim_info_path)

    dsi_total_offset = pd.Timedelta(clock_difference + dsi_fine_offset, unit='s')
    # print(f'DSI offset = {clock_difference} + {dsi_fine_offset:.3f} = {dsi_total_offset.total_seconds():.3f}')
    emg_df = pd.read_parquet(folder_path / f"{emg_block_name}_emg.parquet")
    emg_df.index = emg_df.index + dsi_total_offset
    if scale_emg:
        emg_df.loc[:, :] = scale(emg_df)
    envelope_df = pd.DataFrame(
        signal.sosfiltfilt(filterCoeffs, emg_df.abs().to_numpy(), axis=0),
        index=emg_df.index, columns=emg_df.columns
        )

    left_sweep = -500e-3
    right_sweep = 500e-3
    samples_left = int(left_sweep / emg_sample_interval_sec)
    samples_right = int(right_sweep / emg_sample_interval_sec)
    t = np.arange(samples_left, samples_right) * emg_sample_interval_sec
    num_samples = t.shape[0]

    epoched_emg_dict = {}
    epoched_envelope_dict = {}
    epoch_labels = ['timestamp', 'eid', 'amp', 'freq', 'pw', 'train_idx', 'rank_in_train']
    for timestamp, group in stim_info.groupby('timestamp'):
        key = tuple(group.reset_index().loc[0, epoch_labels])
        first_index = np.flatnonzero(emg_df.index >= timestamp)[0]
        #
        epoched_emg_dict[key] = emg_df.iloc[first_index + samples_left:first_index + samples_right, :].copy()
        epoched_emg_dict[key].index = t
        means_this_trial = epoched_emg_dict[key].mean()
        epoched_emg_dict[key] = epoched_emg_dict[key] - means_this_trial
        #
        epoched_envelope_dict[key] = envelope_df.iloc[first_index + samples_left:first_index + samples_right, :].copy()
        epoched_envelope_dict[key].index = t
        epoched_envelope_dict[key] = epoched_envelope_dict[key] - means_this_trial

    epoched_emg_df = pd.concat(epoched_emg_dict, names=epoch_labels + ['t'])
    epoched_emg_df.columns.name = 'channel'

    file_name_suffix = '_per_pulse' if per_pulse else ''
    print(f'Saving {file_name}_epoched_emg{file_name_suffix}.parquet')
    epoched_emg_df.to_parquet(
        folder_path / (file_name + f'_epoched_emg{file_name_suffix}.parquet'), engine='fastparquet')

    epoched_envelope_df = pd.concat(epoched_envelope_dict, names=epoch_labels + ['t'])
    epoched_envelope_df.columns.name = 'channel'
    epoched_envelope_df.to_parquet(
        folder_path / (file_name + f'_epoched_envelope{file_name_suffix}.parquet'), engine='fastparquet')
