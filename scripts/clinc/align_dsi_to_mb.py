#!/usr/bin/env /users/rdarie/anaconda/isi_analysis/bin/python

import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from isicpy.clinc_lookup_tables import clinc_sample_rate, emg_sample_rate, dsi_trig_sample_rate
from scipy import signal
import numpy as np
from sklearn.preprocessing import minmax_scale
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set(
    context='notebook', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True,
    )

from matplotlib import pyplot as plt

'''filterOpts = {
    'high': {
        'Wn': 1000.,
        'N': 2,
        'btype': 'high',
        'ftype': 'butter'
    },
}'''


def reindex_and_interpolate(df, new_index):
    return df.reindex(df.index.union(new_index)).interpolate(method='index', limit_direction='both').loc[new_index]

def sanitize_triggers(
        srs, fill_val=0, thresh_opts=dict(), plotting=False):
    thresh_opts['plotting'] = plotting
    temp = srs.reset_index(drop=True)
    cross_index, cross_mask = getThresholdCrossings(temp, **thresh_opts)
    cross_mask.iloc[0] = True
    cross_mask.iloc[-1] = True
    cross_timestamps = pd.Series(srs.index[cross_mask])
    durations = pd.Series(cross_timestamps.diff().to_numpy(), index=cross_timestamps)
    valid_duration = pd.Series(False, index=cross_timestamps)
    fudge_factor = 0.05
    for target_duration in [pd.Timedelta(1, unit='s'), pd.Timedelta(2, unit='s')]:
        this_mask = (
            (durations > target_duration * (1 - fudge_factor)) &
            (durations < target_duration * (1 + fudge_factor))
            )
        valid_duration = valid_duration | this_mask
    valid_duration = reindex_and_interpolate(valid_duration, srs.index).fillna(method='ffill')
    if plotting:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(srs)
        ax[1].plot(valid_duration)
        plt.show()
    srs.loc[~valid_duration] = fill_val
    return srs


clinc_sample_interval_sec = float(clinc_sample_rate ** -1)
thresh_opts = dict(
    thresh=0.5, fs=1000, iti=None, absVal=False,
    keep_max=False, edgeType='both',
    plot_opts=dict(whichPeak=10, nSec=12))
plotting = True


folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")

routing_config_info = pd.read_json(folder_path / 'analysis_metadata/routing_config_info.json')

with open(folder_path / 'analysis_metadata/dsi_block_lookup.json', 'r') as f:
    emg_block_lookup = json.load(f)
'''
with open(folder_path / 'analysis_metadata/general_metadata.json', 'r') as f:
    general_metadata = json.load(f)
    file_name_list = general_metadata['file_name_list']
'''

file_name_list = routing_config_info['child_file_name'].to_list()
fine_offsets = {}

previous_parent_file, previous_emg_file, previous_optimal_lag = None, None, None

for file_name in file_name_list:
    if file_name not in emg_block_lookup:
        continue
    emg_block_list = emg_block_lookup[file_name]
    print(f'Synchronizing {file_name}...')

    this_routing_info = routing_config_info.loc[routing_config_info['child_file_name'] == file_name, :]
    parent_file_name = this_routing_info['parent_file_name'].iloc[0]
    print(f'\tIts parent CLINC file is {parent_file_name}')

    fine_offsets[file_name] = {}
    clinc_trigs = pd.read_parquet(folder_path / (file_name + '_clinc_trigs.parquet'))['sync_wave']

    old_tmin, old_tmax = clinc_trigs.index[0], clinc_trigs.index[-1]
    downsampled_clinc_time = (
            old_tmin.round(freq='L') + pd.timedelta_range(
                0, old_tmax - old_tmin,
                freq=pd.Timedelta(1, unit='ms')
                )
            )
    clinc_orig = clinc_trigs.copy()
    clinc_trigs = reindex_and_interpolate(clinc_trigs, downsampled_clinc_time)

    '''
    fig, ax = plt.subplots()
    window_len = pd.Timedelta(12, unit='s')
    plot_mask = clinc_orig.index < clinc_orig.index[0] + window_len
    ax.plot(clinc_orig.loc[plot_mask])
    plot_mask = clinc_trigs.index < clinc_trigs.index[0] + window_len
    ax.plot(clinc_trigs)
    plt.show()
    '''

    for emg_block_name in emg_block_list:
        print(f'\tOn EMG {emg_block_name}...')
        if previous_parent_file is not None:
            if (previous_parent_file == parent_file_name) and (previous_emg_file == emg_block_name):
                # if we're aligning the same parent to the same emg block, we can just reuse the previous offset
                fine_offsets[file_name][emg_block_name] = previous_optimal_lag
                print(f"\t\tReused lag of {int(previous_optimal_lag * 1e3)} samples.")
                continue

        dsi_analog = pd.read_parquet(folder_path / f"{emg_block_name}_dsi_trigs.parquet")
        dsi_trigs = dsi_analog.iloc[:, 1].copy()
        dsi_bounds = dsi_trigs.quantile([5e-2, 95e-2])
        dsi_trigs = (dsi_trigs - dsi_bounds[5e-2]) / (dsi_bounds[95e-2] - dsi_bounds[5e-2])

        tmin = max(dsi_trigs.index[0], clinc_trigs.index[0])
        tmax = min(dsi_trigs.index[-1], clinc_trigs.index[-1])

        masked_clinc = clinc_trigs.loc[(clinc_trigs.index >= tmin) & (clinc_trigs.index <= tmax)]
        masked_dsi = dsi_trigs.loc[(dsi_trigs.index >= tmin) & (dsi_trigs.index <= tmax)]
        if masked_dsi.max() - masked_dsi.min() < 1e-2:
            print('\t\tWarning! DSI channel appears disconnected; Setting fine offset to 0 for manual correction')
            # plt.plot(masked_dsi)
            fine_offsets[file_name][emg_block_name] = 0
            continue
        lags = signal.correlation_lags(masked_clinc.shape[0], masked_dsi.shape[0], mode='full')
        xcorr = signal.correlate(masked_clinc.astype(int).to_numpy(), masked_dsi.astype(int).to_numpy(), mode='full')

        mask = (lags > -1501) & (lags < 1501)
        xcorr_srs = pd.Series(xcorr[mask], index=lags[mask])
        optimal_lag_samples = xcorr_srs.idxmax()
        print(f"\t\tthe optimal lag is {optimal_lag_samples} samples.")

        optimal_lag = optimal_lag_samples * 1e-3
        fine_offsets[file_name][emg_block_name] = optimal_lag
        
        previous_parent_file = parent_file_name
        previous_emg_file = emg_block_name
        previous_optimal_lag = optimal_lag

        if plotting:
            fig, ax = plt.subplots()
            ax.plot(xcorr_srs)
            ax.plot(optimal_lag_samples, xcorr_srs.loc[optimal_lag_samples], 'r*')
            ax.set_title(f'{file_name} synched to {emg_block_name}')
            #
            fig, ax = plt.subplots()
            ax.plot(masked_clinc, lw=1.5, label='clinc_trigs')
            plot_dsi = masked_dsi.copy()
            plot_dsi.index = plot_dsi.index + pd.Timedelta(optimal_lag, unit='s')
            ax.plot(plot_dsi, '--', lw=2, label='dsi_trigs')
            ax.legend(loc='upper right')
            ax.set_title(f'{file_name} synched to {emg_block_name}')
            plt.show()
        print('\t\tDone')

    with open(folder_path / 'analysis_metadata/dsi_to_mb_fine_offsets.json', 'w') as f:
        json.dump(fine_offsets, f, indent=4)
