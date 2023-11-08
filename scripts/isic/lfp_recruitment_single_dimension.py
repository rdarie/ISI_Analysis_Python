
import matplotlib as mpl
import os

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

if 'CCV_HEADLESS' in os.environ:
    mpl.use('Agg')   # generate postscript output
else:
    mpl.use('QT5Agg')   # generate interactive output

import traceback
from isicpy.utils import load_synced_mat, makeFilterCoeffsSOS
from isicpy.lookup_tables import emg_montages, muscle_names
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
        "font.size": 10,
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
    'figure.titlesize': 14,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
sns.set(
    context='talk', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=2, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    mpl.rcParams[rcK] = rcV

# folder_name = "Day2_AM"
# blocks_list = [3]
# this_lfp_montage = emg_montages['lower']
# folder_name = "Day12_PM"
# blocks_list = [4]
folder_name = "Day11_PM"
blocks_list = [2, 3]
this_lfp_montage = emg_montages['lower_v2']
# folder_name = "Day8_AM"
# blocks_list = [1, 2, 3, 4]


data_path = Path(f"/users/rdarie/scratch/3_Preprocessed_Data/{folder_name}")
pdf_folder = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/5_Figures/{folder_name}")
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)

blocks_list_str = '_'.join(f"{block_idx}" for block_idx in blocks_list)

x_axis_name = 'freq_late'
if x_axis_name == 'freq':
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_lfp_recruitment_freq.pdf")
    left_sweep = 0
    right_sweep = int(0.4 * 1e6)
    amp_cutoff = 10e3
elif x_axis_name == 'freq_late':
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_lfp_recruitment_freq_late.pdf")
    left_sweep = int(0.1 * 1e6)
    right_sweep = int(0.4 * 1e6)
    amp_cutoff = 10e3
elif x_axis_name == 'amp':
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_lfp_recruitment_amp.pdf")
    left_sweep = 0
    right_sweep = int(0.1 * 1e6)
    freq_cutoff = 10

filterOptsNotch = {
    'line_noise': {
        'Wn': 60.,
        'nHarmonics': 1,
        'Q': 35,
        'N': 4,
        'btype': 'bandstop',
        'ftype': 'butter'
    },
}

filterOptsPost = {
    'low': {
        'Wn': 100.,
        'N': 4,
        'btype': 'low',
        'ftype': 'butter'
    },
}

verbose = 2
standardize_lfp = True
normalize_across = False

all_stim_info = {}
all_aligned_lfp = {}

parquet_folder = data_path / "parquets"
reprocess_raw = False
save_parquets = True

for block_idx in blocks_list:
    lfp_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_lfp_df.parquet"
    stim_info_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_stim_info_df.parquet"
    file_path = data_path / f"Block{block_idx:0>4d}_Synced_Session_Data.mat"
    if (not os.path.exists(lfp_parquet_path)) or reprocess_raw:
        data_dict = load_synced_mat(
            file_path,
            load_stim_info=True, split_trains=True, stim_info_traces=False, force_trains=True,
            load_vicon=True, vicon_as_df=True,
            load_ripple=True, ripple_variable_names=['NEV'], ripple_as_df=True,
        )
        if data_dict['vicon'] is not None:
            this_lfp = data_dict['vicon']['EMG'].copy()
            this_lfp.rename(columns=this_lfp_montage, inplace=True)
            this_lfp.drop(columns=['NA'], inplace=True)
        if data_dict['stim_info'] is not None:
            all_stim_info[block_idx] = data_dict['stim_info']

            if save_parquets:
                if not os.path.exists(parquet_folder):
                    os.makedirs(parquet_folder)
                this_lfp.to_parquet(lfp_parquet_path)
                all_stim_info[block_idx].to_parquet(stim_info_parquet_path)
    else:
        this_lfp = pd.read_parquet(lfp_parquet_path)
        all_stim_info[block_idx] = pd.read_parquet(stim_info_parquet_path)

    if standardize_lfp:
        lfp_scaler_path = data_path / "pickles" / "lfp_scaler.p"
        with open(lfp_scaler_path, 'rb') as handle:
            scaler = pickle.load(handle)
        this_lfp.loc[:, :] = scaler.transform(this_lfp)

    align_timestamps = all_stim_info[block_idx].index.get_level_values('timestamp_usec')
    aligned_dfs = {}
    analog_time_vector = np.asarray(this_lfp.index)
    nominal_dt = np.int64(np.median(np.diff(analog_time_vector)))
    lfp_sample_rate = np.round((nominal_dt * 1e-6) ** -1)
    epoch_t = np.arange(left_sweep, right_sweep, nominal_dt)
    nominal_num_samp = epoch_t.shape[0]

    filterCoeffsNotch = makeFilterCoeffsSOS(filterOptsNotch.copy(), lfp_sample_rate)
    this_lfp = pd.DataFrame(
        signal.sosfiltfilt(filterCoeffsNotch, this_lfp, axis=0),
        index=this_lfp.index, columns=this_lfp.columns)

    if len(filterOptsPost):
        filterCoeffsPost = makeFilterCoeffsSOS(filterOptsPost.copy(), lfp_sample_rate)
        this_lfp = pd.DataFrame(
            signal.sosfiltfilt(filterCoeffsPost, (this_lfp - this_lfp.mean()).abs(), axis=0),
            index=this_lfp.index, columns=this_lfp.columns)
    else:
        this_lfp = (this_lfp - this_lfp.mean()).abs()

    print(f'Epoching EMG from \n\t{file_path}')
    for timestamp in tqdm(align_timestamps.to_numpy()):
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
        aligned_dfs[timestamp] = pd.DataFrame(
            this_lfp.loc[this_mask, :].to_numpy(),
            index=epoch_t, columns=this_lfp.columns)
    all_aligned_lfp[block_idx] = pd.concat(aligned_dfs, names=['timestamp_usec', 'time_usec'])

stim_info_df = pd.concat(all_stim_info, names=['block', 'timestamp_usec'])
stim_info_df.loc[:, 'elecConfig_str'] = stim_info_df.apply(lambda x: f'-{x["elecCath"]}+{x["elecAno"]}', axis='columns')

reversed_config_strs = stim_info_df.apply(lambda x: f'-{x["elecAno"]}+{x["elecCath"]}', axis='columns').unique().tolist()
config_strs = stim_info_df['elecConfig_str'].unique().tolist()
config_lookup = {
    pair[0]: pair[1]
    for pair in zip(config_strs, reversed_config_strs)
    }

electrode_pairs = []
reordered_elecs = []
orientation_types = {}
parent_elec_configurations = {}
for a, b in config_lookup.items():
    if a in config_strs and b in config_strs:
        idx_a = config_strs.index(a)
        elec_a = config_strs.pop(idx_a)
        idx_b = config_strs.index(b)
        elec_b = config_strs.pop(idx_b)
        electrode_pairs.append((elec_a, elec_b))
        reordered_elecs += [elec_a, elec_b]
        parent_elec_configurations[elec_a] = elec_a
        parent_elec_configurations[elec_b] = elec_a
        orientation_types[elec_a] = 'right side up'
        orientation_types[elec_b] = 'flipped'

def reorder_fun(config_strings):
    return pd.Index([reordered_elecs.index(name) for name in config_strings], name=config_strings.name)

lfp_df = pd.concat(all_aligned_lfp, names=['block', 'timestamp_usec', 'time_usec'])
lfp_df.drop(['L Forearm', 'R Forearm', 'Sync'], axis='columns', inplace=True)

'''
del aligned_dfs, all_aligned_lfp, all_stim_info
gc.collect()
g = sns.displot(data=stim_info_df, x='delta_timestamp_usec', rug=True, element='step', fill=False)
plt.show()
'''
lfp_metadata = lfp_df.index.to_frame()
recruitment_keys = ['elecConfig_str', 'amp', 'freq']

for meta_key in recruitment_keys:
    lfp_metadata.loc[:, meta_key] = lfp_df.index.copy().droplevel('time_usec').map(stim_info_df[meta_key]).to_numpy()
lfp_df.index = pd.MultiIndex.from_frame(lfp_metadata)

#### outlier removal
auc_per_trial = lfp_df.groupby(['block', 'timestamp_usec']).mean()
auc_bar, auc_std = np.mean(auc_per_trial.to_numpy().flatten()), np.std(auc_per_trial.to_numpy().flatten())
n_std = 9
outlier_bounds = (auc_bar - n_std * auc_std, auc_bar + n_std * auc_std)
outlier_mask_per_trial = (auc_per_trial < outlier_bounds[0]) | (auc_per_trial > outlier_bounds[1])
outlier_mask_per_trial = outlier_mask_per_trial.any(axis='columns')
outlier_trials = outlier_mask_per_trial.index[outlier_mask_per_trial]
outlier_mask = pd.MultiIndex.from_frame(lfp_metadata.loc[:, ['block', 'timestamp_usec']]).isin(outlier_trials)
#
lfp_df = lfp_df.loc[~outlier_mask, :]
####
if x_axis_name in ['freq', 'freq_late']:
    # remove amp <= cutoff
    stim_info_df = stim_info_df.loc[stim_info_df['amp'] >= amp_cutoff, :]
    lfp_df = lfp_df.loc[lfp_df.index.get_level_values('amp') >= amp_cutoff, :]
elif x_axis_name == 'amp':
    # remove freq >= cutoff
    stim_info_df = stim_info_df.loc[stim_info_df['freq'] <= freq_cutoff, :]
    lfp_df = lfp_df.loc[lfp_df.index.get_level_values('freq') <= freq_cutoff, :]

#
auc_df = lfp_df.groupby(recruitment_keys + ['block', 'timestamp_usec']).mean()
# temp_average_auc = auc_df.groupby(recruitment_keys).mean()

if normalize_across:
    scaler = MinMaxScaler()
    scaler.fit(auc_df.stack().to_frame())
    auc_df = auc_df.apply(lambda x: scaler.transform(x.reshape(-1, 1)).flatten(), raw=True, axis='index')
else:
    scaler = MinMaxScaler()
    scaler.fit(auc_df)
    auc_df.loc[:, :] = scaler.transform(auc_df)

average_auc_df = auc_df.groupby(recruitment_keys).mean()
temp_average_auc = auc_df.groupby(recruitment_keys).mean()


should_plot_delta_auc = False
if should_plot_delta_auc:
    delta_auc_dict = {}
    for elec_a, elec_b in electrode_pairs:
        auc_a = average_auc_df.xs(elec_a, axis='index', level='elecConfig_str')
        auc_b = average_auc_df.xs(elec_b, axis='index', level='elecConfig_str')
        delta_auc_dict[elec_a] = auc_a - auc_b
    delta_auc_df = pd.concat(delta_auc_dict, names=['elecConfig_str'])

determine_side = lambda x: 'L.' if x[0] == 'L' else 'R.'
if folder_name == 'Day11_PM':
    auc_df.sort_index(level='elecConfig_str', key=reorder_fun, inplace=True)
    average_auc_df.sort_index(level='elecConfig_str', key=reorder_fun, inplace=True)
    if should_plot_delta_auc:
        delta_auc_df.sort_index(level='elecConfig_str', key=reorder_fun, inplace=True)

show_plots = True
with PdfPages(pdf_path) as pdf:
    plot_auc = auc_df.stack().to_frame(name='signal').reset_index()
    plot_auc.loc[:, 'side'] = plot_auc['label'].apply(determine_side)
    plot_auc.loc[:, 'muscle'] = plot_auc['label'].map(muscle_names)

    plot_auc.loc[:, 'parent_elecConfig'] = plot_auc['elecConfig_str'].map(parent_elec_configurations)
    plot_auc.loc[:, 'elec_orientation'] = plot_auc['elecConfig_str'].map(orientation_types)
    if should_plot_delta_auc:
        plot_delta_auc = delta_auc_df.stack().to_frame(name='signal').reset_index()
        plot_delta_auc.loc[:, 'side'] = plot_delta_auc['label'].apply(determine_side)
        plot_delta_auc.loc[:, 'muscle'] = plot_delta_auc['label'].map(muscle_names)
    ###
    elec_subset = plot_auc['elecConfig_str'].unique().tolist()  #  ['-(2,)+(3,)', '-(3,)+(2,)',]
    label_subset = ['LVL', 'LMH', 'LTA', 'LMG', 'LSOL', 'RLVL', 'RMH', 'RTA', 'RMG', 'RSOL']  #  plot_auc['label'].unique().tolist()
    ###
    elec_mask = plot_auc['elecConfig_str'].isin(elec_subset)
    label_mask = plot_auc['label'].isin(label_subset)
    plot_mask = elec_mask & label_mask

    if x_axis_name in ['freq', 'freq_late']:
        g = sns.relplot(
            data=plot_auc.loc[plot_mask, :],
            row='elecConfig_str', col='side',
            hue='muscle',
            x='freq', y='signal', kind='line',
            errorbar='se', linewidth=2
        )
    elif x_axis_name == 'amp':
        g = sns.relplot(
            data=plot_auc.loc[plot_mask, :],
            row='elecConfig_str', col='side',
            hue='muscle',
            x='amp', y='signal', kind='line',
            errorbar='se', linewidth=2
        )
    # change the line width for the legend
    for line in g.legend.get_lines():
        line.set_linewidth(4.0)

    g.figure.suptitle('AUC')
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()
    if x_axis_name in ['freq', 'freq_late']:
        g = sns.relplot(
            data=plot_auc.loc[plot_mask, :],
            row='parent_elecConfig', col='side',
            hue='muscle', style='elec_orientation',
            x='freq', y='signal', kind='line',
            errorbar='se', linewidth=2
        )
    elif x_axis_name == 'amp':
        g = sns.relplot(
            data=plot_auc.loc[plot_mask, :],
            row='parent_elecConfig', col='side',
            hue='muscle', style='elec_orientation',
            x='amp', y='signal', kind='line',
            errorbar='se', linewidth=2
        )
    # change the line width for the legend
    for line in g.legend.get_lines():
        line.set_linewidth(4.0)
    g.figure.suptitle('AUC (orientation pairs)')
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()
    #####
    #####
    if should_plot_delta_auc:
        elec_mask_delta = plot_auc['elecConfig_str'].isin(elec_subset)
        label_mask_delta = plot_auc['label'].isin(label_subset)
        plot_mask_delta = elec_mask_delta & label_mask_delta
        if x_axis_name in ['freq', 'freq_late']:
            g = sns.relplot(
                data=plot_delta_auc.loc[plot_mask_delta, :],
                row='elecConfig_str', col='side',
                hue='muscle',
                x='freq', y='signal', kind='line',
                errorbar=None, linewidth=2
            )
        elif x_axis_name == 'amp':
            g = sns.relplot(
                data=plot_delta_auc.loc[plot_mask_delta, :],
                row='elecConfig_str', col='side',
                hue='muscle',
                x='amp', y='signal', kind='line',
                errorbar=None, linewidth=2
            )
        g.figure.suptitle('delta AUC from dipole rotation')
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if show_plots:
            plt.show()
        else:
            plt.close()
