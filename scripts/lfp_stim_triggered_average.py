
import matplotlib as mpl
import os

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.use('tkagg')   # generate interactive output

import traceback
from isicpy.utils import load_synced_mat, makeFilterCoeffsSOS
from isicpy.lookup_tables import emg_montages
from pathlib import Path
import pandas as pd
import numpy as np
import cloudpickle as pickle
import pdb
import gc
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_pdf import PdfPages

useDPI = 72
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
        "axes.labelsize": 14,
        "axes.titlesize": 18,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 14,
        "legend.title_fontsize": 18,
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
    'figure.titlesize': 16,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
sns.set(
    context='talk', style='white',
    palette='dark', font='sans-serif',
    font_scale=2, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    mpl.rcParams[rcK] = rcV


folder_name = "Day11_PM"
data_path = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/3_Preprocessed_Data/{folder_name}")
parquet_folder = data_path / "parquets"

blocks_list = [2, 3]
blocks_list = [2]
this_emg_montage = emg_montages['lower_v2']
blocks_list_str = '_'.join(f"{block_idx}" for block_idx in blocks_list)

pdf_folder = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/5_Figures/{folder_name}")
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)
pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_lfp_stim_triggered_average.pdf")

this_emg_montage = emg_montages['lower_v2']

left_sweep = - int(50e3)
right_sweep = int(0.1 * 1e6)
verbose = 0

lfp_sample_rate = 15000

filterOpts = {
    'low': {
        'Wn': 500.,
        'N': 4,
        'btype': 'low',
        'ftype': 'butter'
    },
}

subtract_baselines = True
baseline_bounds = [-50e3, 0]
downsample_factor = 1

filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), lfp_sample_rate)

all_stim_info = {}
all_aligned_lfp = {}

for block_idx in blocks_list:
    nf7_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_reref_lfp_df.parquet"
    this_lfp = pd.read_parquet(nf7_parquet_path)
    stim_info_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_stim_info_df.parquet"
    all_stim_info[block_idx] = pd.read_parquet(stim_info_parquet_path)

    align_timestamps = all_stim_info[block_idx].index.get_level_values('timestamp_usec')
    aligned_dfs = {}

    analog_time_vector = np.asarray(this_lfp.index)
    nominal_dt = np.int64(np.median(np.diff(analog_time_vector)))
    lfp_sample_rate = np.round((nominal_dt * 1e-6) ** -1)
    epoch_t = np.arange(left_sweep, right_sweep, nominal_dt)
    nominal_num_samp = epoch_t.shape[0]

    baseline_mask = (epoch_t > baseline_bounds[0]) & (epoch_t < baseline_bounds[1])

    for timestamp in tqdm(align_timestamps.to_numpy()[1:]):
        this_mask = (analog_time_vector >= timestamp + left_sweep)
        first_index = np.flatnonzero(this_mask)[0]

        this_aligned = pd.DataFrame(
            this_lfp.iloc[first_index:first_index + nominal_num_samp, :].to_numpy(),
            index=epoch_t, columns=this_lfp.columns)

        if subtract_baselines:
            baseline = this_aligned.loc[baseline_mask, :].mean()
            this_aligned = (this_aligned - baseline).iloc[::downsample_factor, :]
        else:
            this_aligned = this_aligned.iloc[::downsample_factor, :]

        aligned_dfs[timestamp] = this_aligned
    all_aligned_lfp[block_idx] = pd.concat(aligned_dfs, names=['timestamp_usec', 'time_usec'])

stim_info_df = pd.concat(all_stim_info, names=['block', 'timestamp_usec'])
stim_info_df.loc[:, 'elecConfig_str'] = stim_info_df.apply(lambda x: f'-{x["elecCath"]}+{x["elecAno"]}', axis='columns')

lfp_df = pd.concat(all_aligned_lfp, names=['block', 'timestamp_usec', 'time_usec'])

lfp_metadata = lfp_df.index.to_frame()
recruitment_keys = ['elecConfig_str', 'amp', 'freq']
for meta_key in recruitment_keys:
    lfp_metadata.loc[:, meta_key] = lfp_df.index.copy().droplevel('time_usec').map(stim_info_df[meta_key]).to_numpy()
lfp_df.index = pd.MultiIndex.from_frame(lfp_metadata)

#### outlier removal
auc_per_trial = lfp_df.groupby(['block', 'timestamp_usec']).mean()
auc_bar, auc_std = np.mean(auc_per_trial.to_numpy().flatten()), np.std(auc_per_trial.to_numpy().flatten())

sns.displot(auc_per_trial.to_numpy().flatten())
plt.show()

n_std = 6

outlier_bounds = (auc_bar - n_std * auc_std, auc_bar + n_std * auc_std)
outlier_mask_per_trial = (auc_per_trial < outlier_bounds[0]) | (auc_per_trial > outlier_bounds[1])
outlier_mask_per_trial = outlier_mask_per_trial.any(axis='columns')
outlier_trials = outlier_mask_per_trial.index[outlier_mask_per_trial]
outlier_mask = pd.MultiIndex.from_frame(lfp_metadata.loc[:, ['block', 'timestamp_usec']]).isin(outlier_trials)
#
print(f"{outlier_mask_per_trial.sum() / outlier_mask_per_trial.size} samples rejected")

lfp_df = lfp_df.loc[~outlier_mask, :]
####

show_plots = False
with PdfPages(pdf_path) as pdf:
    ###
    plot_lfp = lfp_df.stack().to_frame(name='signal').reset_index()
    elec_subset = plot_lfp['elecConfig_str'].unique().tolist()  # ['-(2,)+(3,)', '-(3,)+(2,)',]
    ###
    plot_lfp.loc[:, 'time_msec'] = plot_lfp['time_usec'] * 1e-3
    block_timestamp_index = pd.MultiIndex.from_frame(plot_lfp.loc[:, ['block', 'timestamp_usec']])

    for meta_key in ['elecConfig_str', 'amp', 'freq']:
        plot_lfp.loc[:, meta_key] = block_timestamp_index.map(stim_info_df[meta_key]).to_numpy()

    downsampled_mask = plot_lfp['time_usec'].isin(plot_lfp['time_usec'].unique()[::1])
    elec_mask = plot_lfp['elecConfig_str'].isin(elec_subset)
    plot_mask = downsampled_mask & elec_mask

    vert_offset = 5e-2 * (plot_lfp['signal'].max() - plot_lfp['signal'].min())
    horz_offset = 0 * (plot_lfp['time_sec'].max() - plot_lfp['time_sec'].min())
    n_offsets = 0
    for name, group in plot_lfp.groupby('freq'):
        plot_lfp.loc[group.index, 'signal'] = group['signal'] + n_offsets * vert_offset
        plot_lfp.loc[group.index, 'time_sec'] = group['time_sec'] + n_offsets * horz_offset
        n_offsets += 1

    vert_offset = 5e-3 * (plot_lfp['signal'].max() - plot_lfp['signal'].min())
    horz_offset = -5e-2 * (plot_lfp['time_sec'].max() - plot_lfp['time_sec'].min())
    n_offsets = 0
    for name, group in plot_lfp.groupby('amp'):
        plot_lfp.loc[group.index, 'signal'] = group['signal'] + n_offsets * vert_offset
        plot_lfp.loc[group.index, 'time_sec'] = group['time_sec'] + n_offsets * horz_offset
        n_offsets += 1

    print('Saving stim-triggered plots')
    for elecConfig in tqdm(elec_subset):
        elec_mask = plot_lfp['elecConfig_str'] == elecConfig
        g = sns.relplot(
            data=plot_lfp.loc[plot_mask & elec_mask, :],
            col='label', col_wrap=5,
            x='time_sec', y='signal',
            hue='amp', style='freq', dashes=False,
            kind='line',
            # units='timestamp_usec', estimator=None,
            errorbar=None,
            height=5, aspect=2 / 3
            )
        g.figure.suptitle(f"{elecConfig}")
        # change the line width for the legend
        for line in g.legend.get_lines():
            line.set_linewidth(4.0)
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if show_plots:
            plt.show()
        else:
            plt.close()
