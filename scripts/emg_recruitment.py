import traceback
from isicpy.utils import load_synced_mat, closestSeries
from isicpy.lookup_tables import emg_montages
from pathlib import Path
import pandas as pd
import numpy as np
import cloudpickle as pickle
import pdb
import os
import gc
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    mpl.use('Agg')   # generate postscript output
else:
    mpl.use('QT5Agg')   # generate interactive output
useDPI = 200
dpiFactor = 72 / useDPI

import seaborn as sns
from matplotlib import pyplot as plt

snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': .5,
        'lines.markersize': 2.5,
        'patch.linewidth': .5,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .2,
        "font.size": 4,
        "axes.labelsize": 7,
        "axes.titlesize": 9,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 7,
        "legend.title_fontsize": 9,
        "xtick.bottom": True,
        "xtick.top": False,
        "ytick.left": True,
        "ytick.right": False,
        "xtick.major.width": .125,
        "ytick.major.width": .125,
        "xtick.minor.width": .125,
        "ytick.minor.width": .125,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.minor.size": 1,
        "ytick.minor.size": 1,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
    }
mplRCParams = {
    'figure.titlesize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
sns.set(
    context='talk', style='white',
    palette='dark', font='sans-serif',
    font_scale=2, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    mpl.rcParams[rcK] = rcV



data_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/3_Preprocessed_Data/Day11_PM")

left_sweep = int(-0.05 * 1e6)
right_sweep = int(0.3 * 1e6)
verbose = 0
standardize_emg = True
if standardize_emg:
    emg_scaler_path = data_path / "pickles" / "emg_scaler.p"
    with open(emg_scaler_path, 'rb') as handle:
        scaler = pickle.load(handle)

all_stim_info = {}
all_aligned_emg = {}
for block_idx in [2, 3]:
    file_path = data_path / f"Block{block_idx:0>4d}_Synced_Session_Data.mat"
    data_dict = load_synced_mat(
        file_path,
        load_stim_info=True,
        load_vicon=True, vicon_as_df=True,
        load_ripple=True, ripple_variable_names=['NEV'], ripple_as_df=True
        )

    if data_dict['vicon'] is not None:
        this_emg = data_dict['vicon']['EMG'].iloc[:, :12].copy()
        this_emg.columns = emg_montages['lower']
        this_emg.columns.name = 'label'

    if data_dict['stim_info'] is not None:
        # data_dict['stim_info'].loc[:, 'elecCath'] = data_dict['stim_info']['elecCath'].apply(lambda x: str(x))
        # data_dict['stim_info'].loc[:, 'elecAno'] = data_dict['stim_info']['elecAno'].apply(lambda x: str(x))
        # align to stim onset
        all_stim_info[block_idx] = data_dict['stim_info'].loc[data_dict['stim_info']['amp'] != 0, :].reset_index(drop=True)
        closest_nev_times, _ = closestSeries(
            referenceIdx=all_stim_info[block_idx]['timestamp_usec'],
            sampleFrom=data_dict['ripple']['NEV']['time_usec'], strictly='greater')
        all_stim_info[block_idx].loc[:, 'original_timestamp_usec'] = all_stim_info[block_idx]['timestamp_usec'].copy()
        all_stim_info[block_idx].loc[:, 'timestamp_usec'] = closest_nev_times.to_numpy()
        all_stim_info[block_idx].loc[:, 'delta_timestamp_usec'] = all_stim_info[block_idx]['original_timestamp_usec'].to_numpy() - closest_nev_times.to_numpy()
        all_stim_info[block_idx].set_index('timestamp_usec', inplace=True)

        aligned_dfs = {}
        analog_time_vector = np.asarray(this_emg.index)
        nominal_dt = np.int64(np.median(np.diff(analog_time_vector)))
        epoch_t = np.arange(left_sweep, right_sweep, nominal_dt)
        nominal_num_samp = epoch_t.shape[0]
        print(f'Epoching EMG from \n\t{file_path}')
        for timestamp in tqdm(closest_nev_times.to_numpy()):
            this_mask = (analog_time_vector >= timestamp + left_sweep) & (analog_time_vector <= timestamp + right_sweep)
            sweep_offset = 0
            while this_mask.sum() != nominal_num_samp:
                # fix malformed epochs caused by floating point comparison errors
                if this_mask.sum() > nominal_num_samp:
                    sweep_offset -= nominal_dt
                else:
                    sweep_offset += nominal_dt
                if verbose > 1:
                    print(f'sweep offset set to {sweep_offset}')
                this_mask = (analog_time_vector >= timestamp + left_sweep - nominal_dt / 2) & (analog_time_vector < timestamp + right_sweep + sweep_offset + nominal_dt / 2)
                if verbose > 1:
                    print(f'this_mask.sum() = {this_mask.sum()}')
            if standardize_emg:
                aligned_dfs[timestamp] = pd.DataFrame(
                    scaler.transform(this_emg.loc[this_mask, :]),
                    index=epoch_t, columns=this_emg.columns)
            else:
                aligned_dfs[timestamp] = pd.DataFrame(
                    this_emg.loc[this_mask, :].to_numpy(),
                    index=epoch_t, columns=this_emg.columns)

        all_aligned_emg[block_idx] = pd.concat(aligned_dfs, names=['timestamp_usec', 'time_usec'])

stim_info_df = pd.concat(all_stim_info, names=['block', 'timestamp_usec'])
stim_info_df.loc[:, 'elecConfig_str'] = stim_info_df.apply(lambda x: f'-{x["elecCath"]}+{x["elecAno"]}', axis='columns')
emg_df = pd.concat(all_aligned_emg, names=['block', 'timestamp_usec', 'time_usec'])

del aligned_dfs, all_aligned_emg, all_stim_info
gc.collect()


g = sns.displot(data=stim_info_df, x='delta_timestamp_usec', rug=True, element='step', fill=False)
plt.show()

plot_emg = emg_df.stack().to_frame(name='signal').reset_index()
plot_emg.loc[:, 'time_sec'] = plot_emg['time_usec'] * 1e-6
block_timestamp_index = pd.MultiIndex.from_frame(plot_emg.loc[:, ['block', 'timestamp_usec']])
for meta_key in ['elecConfig_str', 'amp', 'freq']:
    plot_emg.loc[:, meta_key] = block_timestamp_index.map(stim_info_df[meta_key]).to_numpy()

downsampled_mask = plot_emg['time_usec'].isin(plot_emg['time_usec'].unique()[::1])
channels_mask = plot_emg['label'].isin(['RLVL', 'RMH',])
plot_mask = downsampled_mask

for elecConfig in stim_info_df['elecConfig_str'].unique():
    elec_mask = plot_emg['elecConfig_str'] == elecConfig
    g = sns.relplot(
        data=plot_emg.loc[plot_mask & elec_mask, :],
        col='label', col_wrap=5,
        x='time_sec', y='signal',
        hue='amp', style='freq',
        kind='line',
        # units='timestamp_usec', estimator=None,
        errorbar='sd',
        )
    plt.show()
