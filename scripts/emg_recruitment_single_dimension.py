
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


data_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/3_Preprocessed_Data/Day8_AM")
blocks_list = [1, 2, 3, 4]
blocks_list_str = '_'.join(f"{block_idx}" for block_idx in blocks_list)

x_axis_name = 'amp'

if x_axis_name == 'freq':
    pdf_path = data_path / f"Blocks_{blocks_list_str}_emg_recruitment_freq.pdf"
    left_sweep = 0
    right_sweep = int(0.1 * 1e6)
    amp_cutoff = 9e3
elif x_axis_name == 'amp':
    pdf_path = data_path / f"Blocks_{blocks_list_str}_emg_recruitment_amp.pdf"
    left_sweep = 0
    right_sweep = int(0.1 * 1e6)
    freq_cutoff = 10

verbose = 0
standardize_emg = True

if standardize_emg:
    emg_scaler_path = data_path / "pickles" / "emg_scaler.p"
    with open(emg_scaler_path, 'rb') as handle:
        scaler = pickle.load(handle)

'''filterOpts = {
    'low': {
        'Wn': 500.,
        'N': 4,
        'btype': 'low',
        'ftype': 'butter'
    },
}'''

all_stim_info = {}
all_aligned_emg = {}

for block_idx in blocks_list:
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
        all_stim_info[block_idx] = data_dict['stim_info']
        align_timestamps = all_stim_info[block_idx].index.get_level_values('timestamp_usec')
        aligned_dfs = {}
        analog_time_vector = np.asarray(this_emg.index)
        nominal_dt = np.int64(np.median(np.diff(analog_time_vector)))
        emg_sample_rate = np.round((nominal_dt * 1e-6) ** -1)
        epoch_t = np.arange(left_sweep, right_sweep, nominal_dt)
        nominal_num_samp = epoch_t.shape[0]

        if standardize_emg:
            this_emg.loc[:, :] = scaler.transform(this_emg)

        '''
        filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), emg_sample_rate)
        this_emg = pd.DataFrame(
            signal.sosfiltfilt(filterCoeffs, (this_emg - this_emg.mean()).abs(), axis=0),
            index=this_emg.index, columns=this_emg.columns)
            '''
        this_emg = (this_emg - this_emg.mean()).abs()
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
                this_emg.loc[this_mask, :].to_numpy(),
                index=epoch_t, columns=this_emg.columns)
        all_aligned_emg[block_idx] = pd.concat(aligned_dfs, names=['timestamp_usec', 'time_usec'])

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

emg_df = pd.concat(all_aligned_emg, names=['block', 'timestamp_usec', 'time_usec'])
emg_df.drop(['LMH', 'L Forearm', 'R Forearm'], axis='columns', inplace=True)

'''
del aligned_dfs, all_aligned_emg, all_stim_info
gc.collect()
g = sns.displot(data=stim_info_df, x='delta_timestamp_usec', rug=True, element='step', fill=False)
plt.show()
'''
emg_metadata = emg_df.index.to_frame()
recruitment_keys = ['elecConfig_str', 'amp', 'freq']

### ['block', 'elecConfig_str', 'amp_binned', 'freq_binned']
stim_info_df.loc[:, 'freq_binned'] = pd.cut(stim_info_df['freq'], bins=10)
stim_info_df.loc[:, 'amp_binned'] = pd.cut(stim_info_df['amp'], bins=10)
counts_df = stim_info_df.reset_index().groupby(['block', 'elecConfig_str']).count().iloc[:, 0]
counts_df = counts_df.loc[counts_df != 0]
counts_df.name = 'count'
counts_df = counts_df.reset_index()
counts_df.sort_values(by=['block', 'count'])
###

for meta_key in recruitment_keys:
    emg_metadata.loc[:, meta_key] = emg_df.index.copy().droplevel('time_usec').map(stim_info_df[meta_key]).to_numpy()
emg_df.index = pd.MultiIndex.from_frame(emg_metadata)

#### outlier removal
auc_per_trial = emg_df.groupby(['block', 'timestamp_usec']).mean()
auc_bar, auc_std = np.mean(auc_per_trial.to_numpy().flatten()), np.std(auc_per_trial.to_numpy().flatten())
n_std = 6
outlier_bounds = (auc_bar - n_std * auc_std, auc_bar + n_std * auc_std)
outlier_mask_per_trial = (auc_per_trial < outlier_bounds[0]) | (auc_per_trial > outlier_bounds[1])
outlier_mask_per_trial = outlier_mask_per_trial.any(axis='columns')
outlier_trials = outlier_mask_per_trial.index[outlier_mask_per_trial]
outlier_mask = pd.MultiIndex.from_frame(emg_metadata.loc[:, ['block', 'timestamp_usec']]).isin(outlier_trials)
#
emg_df = emg_df.loc[~outlier_mask, :]
####
if x_axis_name == 'freq':
    # remove amp <= cutoff
    stim_info_df = stim_info_df.loc[stim_info_df['amp'] >= amp_cutoff, :]
    emg_df = emg_df.loc[emg_df.index.get_level_values('amp') >= amp_cutoff, :]
elif x_axis_name == 'amp':
    # remove freq >= cutoff
    stim_info_df = stim_info_df.loc[stim_info_df['freq'] <= freq_cutoff, :]
    emg_df = emg_df.loc[emg_df.index.get_level_values('freq') <= freq_cutoff, :]

#

auc_df = emg_df.groupby(recruitment_keys + ['block', 'timestamp_usec']).mean()
scaler = MinMaxScaler()
auc_df.loc[:, :] = scaler.fit_transform(auc_df)
average_auc_df = auc_df.groupby(recruitment_keys).mean()

delta_auc_dict = {}
for elec_a, elec_b in electrode_pairs:
    auc_a = average_auc_df.xs(elec_a, axis='index', level='elecConfig_str')
    auc_b = average_auc_df.xs(elec_b, axis='index', level='elecConfig_str')
    delta_auc_dict[elec_a] = auc_a - auc_b

delta_auc_df = pd.concat(delta_auc_dict, names=['elecConfig_str'])

determine_side = lambda x: 'Left' if x[0] == 'L' else 'Right'

auc_df.sort_index(level='elecConfig_str', key=reorder_fun, inplace=True)

show_plots = False
with PdfPages(pdf_path) as pdf:
    plot_auc = auc_df.stack().to_frame(name='signal').reset_index()
    plot_auc.loc[:, 'side'] = plot_auc['label'].apply(determine_side)
    plot_auc.loc[:, 'muscle'] = plot_auc['label'].map(muscle_names)

    plot_auc.loc[:, 'parent_elecConfig'] = plot_auc['elecConfig_str'].map(parent_elec_configurations)
    plot_auc.loc[:, 'elec_orientation'] = plot_auc['elecConfig_str'].map(orientation_types)

    plot_delta_auc = delta_auc_df.stack().to_frame(name='signal').reset_index()
    plot_delta_auc.loc[:, 'side'] = plot_delta_auc['label'].apply(determine_side)
    plot_delta_auc.loc[:, 'muscle'] = plot_delta_auc['label'].map(muscle_names)
    ###
    elec_subset = ['-(2,)+(3,)', '-(3,)+(2,)',]  #  plot_auc['elecConfig_str'].unique().tolist()
    label_subset = ['LVL', 'LTA', 'LMG', 'RLVL', 'RTA', 'RMG']  #  plot_auc['label'].unique().tolist()
    ###
    elec_mask = plot_auc['elecConfig_str'].isin(elec_subset)
    label_mask = plot_auc['label'].isin(label_subset)
    plot_mask = elec_mask & label_mask

    g = sns.relplot(
        data=plot_auc.loc[plot_mask, :],
        row='elecConfig_str', col='side',
        hue='muscle',
        x=x_axis_name, y='signal', kind='line',
        errorbar='se', linewidth=2
    )

    g.figure.suptitle('AUC')
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()

    g = sns.relplot(
        data=plot_auc.loc[plot_mask, :],
        row='parent_elecConfig', col='side',
        hue='muscle', style='elec_orientation',
        x=x_axis_name, y='signal', kind='line',
        errorbar='se', linewidth=2
    )
    g.figure.suptitle('AUC (orientation pairs)')
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()
    #####
    #####
    elec_mask_delta = plot_auc['elecConfig_str'].isin(elec_subset)
    label_mask_delta = plot_auc['label'].isin(label_subset)
    plot_mask_delta = elec_mask_delta & label_mask_delta

    g = sns.relplot(
        data=plot_delta_auc.loc[plot_mask_delta, :],
        row='elecConfig_str', col='side',
        hue='muscle',
        x=x_axis_name, y='signal', kind='line',
        errorbar=None, linewidth=2
    )
    g.figure.suptitle('delta AUC from dipole rotation')
    pdf.savefig(bbox_inches='tight', pad_inches=0)

    if show_plots:
        plt.show()
    else:
        plt.close()
