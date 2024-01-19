import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
from matplotlib import pyplot as plt

import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from isicpy.clinc_lookup_tables import clinc_sample_rate, emg_sample_rate, dsi_mb_clock_offsets
from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler

clinc_sample_interval_sec = float(clinc_sample_rate ** -1)

filterOptsClinc = {
    'low': {
        'Wn': 500.,
        'N': 8,
        'btype': 'low',
        'ftype': 'butter'
    },
}
filterCoeffsClinc = makeFilterCoeffsSOS(filterOptsClinc.copy(), clinc_sample_rate)
apply_clinc_filters = True

filterOptsEmg = {
    'low': {
        'Wn': 50.,
        'N': 4,
        'btype': 'low',
        'ftype': 'butter'
    }
}
filterCoeffsEmg = makeFilterCoeffsSOS(filterOptsEmg.copy(), emg_sample_rate)

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")
file_name_list = ["MB_1702049441_627410", "MB_1702049896_129326"]
downsample_factor_lfp = 5
for file_name in file_name_list:
    with open(folder_path / 'dsi_block_lookup.json', 'r') as f:
        emg_block_name = json.load(f)[file_name][0]
    clock_difference = dsi_mb_clock_offsets[folder_path.stem]
    with open(folder_path / 'dsi_to_mb_fine_offsets.json', 'r') as f:
        dsi_fine_offset = json.load(f)[file_name][emg_block_name]

    dsi_total_offset = pd.Timedelta(clock_difference + dsi_fine_offset, unit='s')
    print(f'DSI offset = {clock_difference} + {dsi_fine_offset:.3f} = {dsi_total_offset.total_seconds():.3f}')
    emg_df = pd.read_parquet(folder_path / f"{emg_block_name}_emg.parquet")
    emg_df.index = emg_df.index + dsi_total_offset

    clinc_df = pd.read_parquet(folder_path / (file_name + '_clinc.parquet'))
    if apply_clinc_filters:
        clinc_df = pd.DataFrame(
            signal.sosfiltfilt(filterCoeffsClinc, clinc_df, axis=0),
            index=clinc_df.index, columns=clinc_df.columns)

    reref_df = pd.read_parquet(folder_path / (file_name + '_clinc_reref.parquet'))
    if apply_clinc_filters:
        reref_df = pd.DataFrame(
            signal.sosfiltfilt(filterCoeffsClinc, reref_df, axis=0),
            index=reref_df.index, columns=reref_df.columns)
    envelope_df = pd.DataFrame(
        signal.sosfiltfilt(filterCoeffsEmg, emg_df.abs().to_numpy(), axis=0),
        index=emg_df.index, columns=emg_df.columns
    )
    tens_info = pd.read_parquet(folder_path / (file_name + '_tens_info.parquet'))

    mean_bounds = [-50e-3, 0]

    left_sweep_lfp = -25e-3
    right_sweep_lfp = 100e-3
    samples_left_lfp = int(left_sweep_lfp / clinc_sample_interval_sec)
    samples_right_lfp = int(right_sweep_lfp / clinc_sample_interval_sec)
    t_lfp = np.arange(samples_left_lfp, samples_right_lfp) * clinc_sample_interval_sec
    mean_mask_lfp = (t_lfp >= mean_bounds[0]) & (t_lfp <= mean_bounds[1])
    num_samples_lfp = t_lfp.shape[0]

    left_sweep_emg = -50e-3
    right_sweep_emg = 200e-3
    samples_left_emg = int(left_sweep_emg * emg_sample_rate)
    samples_right_emg = int(right_sweep_emg * emg_sample_rate)
    t_emg = np.arange(samples_left_emg, samples_right_emg) / emg_sample_rate
    mean_mask_emg = (t_emg >= mean_bounds[0]) & (t_emg <= mean_bounds[1])
    num_samples_emg = t_emg.shape[0]

    epoched_dict = {}
    reref_dict = {}
    epoched_emg_dict = {}
    epoched_envelope_dict = {}
    epoch_labels = ['timestamp', 'location', 'amp', 'pw']
    for timestamp, group in tens_info.groupby('timestamp'):
        key = tuple(group.reset_index().loc[0, epoch_labels])
        first_index_lfp = np.flatnonzero(clinc_df.index >= timestamp)[0]

        epoched_dict[key] = clinc_df.iloc[first_index_lfp + samples_left_lfp:first_index_lfp + samples_right_lfp, :].copy()
        epoched_dict[key].index = t_lfp
        means_this_trial = epoched_dict[key].loc[mean_mask_lfp, :].mean()
        epoched_dict[key] = epoched_dict[key] - means_this_trial
        epoched_dict[key] = epoched_dict[key].iloc[::downsample_factor_lfp, :]

        reref_dict[key] = reref_df.iloc[first_index_lfp + samples_left_lfp:first_index_lfp + samples_right_lfp, :].copy()
        reref_dict[key].index = t_lfp
        means_this_trial = reref_dict[key].loc[mean_mask_lfp, :].mean()
        reref_dict[key] = reref_dict[key] - means_this_trial
        reref_dict[key] = reref_dict[key].iloc[::downsample_factor_lfp, :]

        first_index_emg = np.flatnonzero(emg_df.index >= timestamp)[0]
        epoched_emg_dict[key] = emg_df.iloc[first_index_emg + samples_left_emg:first_index_emg + samples_right_emg, :].copy()
        epoched_emg_dict[key].index = t_emg

        means_this_trial = epoched_emg_dict[key].loc[mean_mask_emg, :].mean()
        epoched_emg_dict[key] = epoched_emg_dict[key] - means_this_trial
        #
        epoched_envelope_dict[key] = envelope_df.iloc[first_index_emg + samples_left_emg:first_index_emg + samples_right_emg, :].copy()
        epoched_envelope_dict[key].index = t_emg
        epoched_envelope_dict[key] = epoched_envelope_dict[key] - means_this_trial

    lfp_df = pd.concat(epoched_dict, names=epoch_labels + ['t'])
    lfp_df.columns.name = 'channel'
    lfp_df.to_parquet(
        folder_path / (file_name + f'_tens_epoched_lfp.parquet'), engine='fastparquet')

    reref_lfp_df = pd.concat(reref_dict, names=epoch_labels + ['t'])
    reref_lfp_df.columns.name = 'channel'
    reref_lfp_df.to_parquet(
        folder_path / (file_name + f'_tens_epoched_reref_lfp.parquet'), engine='fastparquet')

    epoched_emg_df = pd.concat(epoched_emg_dict, names=epoch_labels + ['t'])
    epoched_emg_df.columns.name = 'channel'
    epoched_emg_df.to_parquet(
        folder_path / (file_name + f'_tens_epoched_emg.parquet'), engine='fastparquet')

    epoched_envelope_df = pd.concat(epoched_envelope_dict, names=epoch_labels + ['t'])
    epoched_envelope_df.columns.name = 'channel'
    epoched_envelope_df.to_parquet(
        folder_path / (file_name + f'_tens_epoched_envelope.parquet'), engine='fastparquet')
