
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

pd_idx = pd.IndexSlice
useDPI = 200
dpiFactor = 72 / useDPI
font_zoom_factor = 1.
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
        "font.size": 4 * font_zoom_factor,
        "axes.labelsize": 14 * font_zoom_factor,
        "axes.titlesize": 18 * font_zoom_factor,
        "xtick.labelsize": 10 * font_zoom_factor,
        "ytick.labelsize": 10 * font_zoom_factor,
        "legend.fontsize": 14 * font_zoom_factor,
        "legend.title_fontsize": 18 * font_zoom_factor,
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
    'figure.titlesize': 16 * font_zoom_factor,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
sns.set(
    context='talk', style='white',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    mpl.rcParams[rcK] = rcV


folder_name = "Day2_AM"
blocks_list = [3]
this_emg_montage = emg_montages['lower']

# folder_name = "Day11_PM"
# blocks_list = [2, 3]
# this_emg_montage = emg_montages['lower_v2']

data_path = Path(f"/users/rdarie/scratch/3_Preprocessed_Data/{folder_name}")
blocks_list_str = '_'.join(f"{block_idx}" for block_idx in blocks_list)

pdf_folder = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/5_Figures/{folder_name}")
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)

filterOptsNotch = {
    'line_noise': {
        'Wn': 60.,
        'nHarmonics': 1,
        'Q': 35,
        'N': 12,
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

x_axis_name = 'amp'
left_sweep = - int(100 * 1e3)
right_sweep = int(500 * 1e3)
verbose = 0
plots_dt = 2e-3
if x_axis_name == 'freq':
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_emg_sta_freq.pdf")
    window_left_sweep = 0
    window_right_sweep = int(0.4 * 1e6)
    amp_cutoff = 9e3
elif x_axis_name == 'freq_late':
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_emg_sta_freq_late.pdf")
    window_left_sweep = int(0.1 * 1e6)
    window_right_sweep = int(0.4 * 1e6)
    amp_cutoff = 9e3
elif x_axis_name == 'amp':
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_emg_sta_amp_{filterOptsNotch['line_noise']['N']}_taps_Q_{filterOptsNotch['line_noise']['Q']}_{filterOptsNotch['line_noise']['nHarmonics']}_harmonics.pdf")
    window_left_sweep = 0
    window_right_sweep = int(0.1 * 1e6)
    freq_cutoff = 10

standardize_emg = True
if standardize_emg:
    emg_scaler_path = data_path / "pickles" / "emg_scaler.p"
    with open(emg_scaler_path, 'rb') as handle:
        scaler = pickle.load(handle)

all_stim_info = {}
all_aligned_emg = {}

parquet_folder = data_path / "parquets"
reprocess_raw = False
save_parquets = True

for block_idx in blocks_list:
    emg_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_emg_df.parquet"
    stim_info_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_stim_info_df.parquet"
    file_path = data_path / f"Block{block_idx:0>4d}_Synced_Session_Data.mat"

    if (not os.path.exists(emg_parquet_path)) or reprocess_raw:
        data_dict = load_synced_mat(
            file_path,
            load_stim_info=True,
            load_vicon=True, vicon_as_df=True,
            load_ripple=True, ripple_variable_names=['NEV'], ripple_as_df=True
            )

        if data_dict['vicon'] is not None:
            this_emg = data_dict['vicon']['EMG'].copy()
            this_emg.rename(columns=this_emg_montage, inplace=True)
            this_emg.drop(columns=['NA'], inplace=True)

        if data_dict['stim_info'] is not None:
            all_stim_info[block_idx] = data_dict['stim_info']

            if standardize_emg:
                this_emg.loc[:, :] = scaler.transform(this_emg)

            if save_parquets:
                if not os.path.exists(parquet_folder):
                    os.makedirs(parquet_folder)
                this_emg.to_parquet(emg_parquet_path)
                all_stim_info[block_idx].to_parquet(stim_info_parquet_path)
    else:
        this_emg = pd.read_parquet(emg_parquet_path)
        all_stim_info[block_idx] = pd.read_parquet(stim_info_parquet_path)

    align_timestamps = all_stim_info[block_idx].index.get_level_values('timestamp_usec')
    aligned_dfs = {}

    analog_time_vector = np.asarray(this_emg.index)
    nominal_dt = np.int64(np.median(np.diff(analog_time_vector)))
    emg_sample_rate = np.round((nominal_dt * 1e-6) ** -1)
    emg_downsample = int(np.ceil(plots_dt / nominal_dt))
    epoch_t = np.arange(left_sweep, right_sweep, nominal_dt)
    nominal_num_samp = epoch_t.shape[0]

    filterCoeffsNotch = makeFilterCoeffsSOS(filterOptsNotch.copy(), emg_sample_rate)
    this_emg = pd.DataFrame(
        signal.sosfiltfilt(filterCoeffsNotch, this_emg, axis=0),
        index=this_emg.index, columns=this_emg.columns)

    if len(filterOptsPost):
        filterCoeffsPost = makeFilterCoeffsSOS(filterOptsPost.copy(), emg_sample_rate)
        this_emg = pd.DataFrame(
            signal.sosfiltfilt(filterCoeffsPost, (this_emg - this_emg.mean()).abs(), axis=0),
            index=this_emg.index, columns=this_emg.columns)
    else:
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
for a, b in config_lookup.items():
    if a in config_strs and b in config_strs:
        idx_a = config_strs.index(a)
        elec_a = config_strs.pop(idx_a)
        idx_b = config_strs.index(b)
        elec_b = config_strs.pop(idx_b)
        electrode_pairs.append((elec_a, elec_b))
        reordered_elecs += [elec_a, elec_b]
        orientation_types[elec_a] = 'right side up'
        orientation_types[elec_b] = 'flipped'

def reorder_fun(config_strings):
    return pd.Index([reordered_elecs.index(name) for name in config_strings], name=config_strings.name)

emg_df = pd.concat(all_aligned_emg, names=['block', 'timestamp_usec', 'time_usec'])
emg_df.drop(['L Forearm', 'R Forearm', 'Sync'], axis='columns', inplace=True)

'''
del aligned_dfs, all_aligned_emg, all_stim_info
gc.collect()
g = sns.displot(data=stim_info_df, x='delta_timestamp_usec', rug=True, element='step', fill=False)
plt.show()
'''

emg_metadata = emg_df.index.to_frame()
recruitment_keys = ['elecConfig_str', 'amp', 'freq']
for meta_key in recruitment_keys:
    emg_metadata.loc[:, meta_key] = emg_df.index.copy().droplevel('time_usec').map(stim_info_df[meta_key]).to_numpy()
emg_df.index = pd.MultiIndex.from_frame(emg_metadata)

#### outlier removal
auc_per_trial = emg_df.groupby(['block', 'timestamp_usec']).mean()
auc_bar, auc_std = np.mean(auc_per_trial.to_numpy().flatten()), np.std(auc_per_trial.to_numpy().flatten())
n_std = 12
outlier_bounds = (auc_bar - n_std * auc_std, auc_bar + n_std * auc_std)
outlier_mask_per_trial = (auc_per_trial < outlier_bounds[0]) | (auc_per_trial > outlier_bounds[1])
outlier_mask_per_trial = outlier_mask_per_trial.any(axis='columns')
outlier_trials = outlier_mask_per_trial.index[outlier_mask_per_trial]
outlier_mask = pd.MultiIndex.from_frame(emg_metadata.loc[:, ['block', 'timestamp_usec']]).isin(outlier_trials)
emg_df = emg_df.loc[~outlier_mask, :]
####
if folder_name == 'Day11_PM':
    emg_df.sort_index(level='elecConfig_str', key=reorder_fun, inplace=True)

## detrend
# for name, group in emg_df.groupby(['block', 'timestamp_usec']):
#     emg_df.loc[group.index, :] = group - group.mean()

show_plots = False
with PdfPages(pdf_path) as pdf:
    ###
    if x_axis_name in ['freq', 'freq_late']:
        recruitment_mask = emg_df.index.get_level_values('amp') >= amp_cutoff
        hue_var = 'freq'
    if x_axis_name == 'amp':
        recruitment_mask = emg_df.index.get_level_values('freq') <= freq_cutoff
        hue_var = 'amp'
    ###
    # plot_emg = emg_df.loc[recruitment_mask, :].groupby(
    #     ['elecConfig_str', hue_var, 'time_usec']).mean().stack().to_frame(name='signal').reset_index()
    plot_emg = emg_df.loc[recruitment_mask, :].stack().to_frame(name='signal').reset_index()
    plot_emg.loc[:, 'time_sec'] = plot_emg['time_usec'] * 1e-6
    downsampled_mask = plot_emg['time_usec'].isin(plot_emg['time_usec'].unique()[::emg_downsample])

    if x_axis_name in ['freq', 'freq_late']:
        elec_subset = plot_emg['elecConfig_str'].unique().tolist()   #  ['-(18,)+(26,)', '-(26,)+(27,)', '-(18,)+(27,)']
        label_subset = plot_emg['label'].unique().tolist()  # ['RTA', 'RMG', 'RSOL']
        hue_var = 'freq'
    if x_axis_name == 'amp':
        elec_subset = plot_emg['elecConfig_str'].unique().tolist()   #  ['-(18,)+(26,)', '-(26,)+(27,)', '-(18,)+(27,)']
        label_subset = plot_emg['label'].unique().tolist()  #  ['RTA', 'RMG', 'RSOL']

    label_mask = plot_emg['label'].isin(label_subset)
    elec_mask = plot_emg['elecConfig_str'].isin(elec_subset)
    plot_mask = elec_mask & label_mask #  & downsampled_mask
    plot_emg = plot_emg.loc[plot_mask, :].copy()

    hline_df = plot_emg.loc[:, ['elecConfig_str', hue_var, 'block', 'timestamp_usec', 'signal']].groupby(['elecConfig_str', hue_var, 'block', 'timestamp_usec']).mean() * 0

    vert_span = plot_emg.loc[:, 'signal'].max() - plot_emg.loc[:, 'signal'].min()
    horz_span = plot_emg.loc[:, 'time_sec'].max() - plot_emg.loc[:, 'time_sec'].min()

    vert_offset = 10e-2 * vert_span
    horz_offset = 0 * horz_span
    vert_sub_offset = 5e-2 * vert_span
    horz_sub_offset = 0 * horz_span

    for row_name, row_group in plot_emg.groupby('elecConfig_str'):
        total_vert_offset = 0
        total_horz_offset = 0
        for hue_name, hue_group in row_group.groupby(hue_var):
            # print(f'hue_group.shape = {hue_group.shape}')
            for sub_name, sub_group in hue_group.groupby(['block', 'timestamp_usec']):
                # print(f'\tsub_group.shape = {sub_group.shape}')
                # print(f'\n{sub_group}')
                plot_emg.loc[sub_group.index, 'signal'] += total_vert_offset
                hline_df.loc[pd_idx[row_name, hue_name, sub_name[0], sub_name[1]], :] += total_vert_offset
                plot_emg.loc[sub_group.index, 'time_sec'] += total_horz_offset
                total_vert_offset += vert_sub_offset
                total_horz_offset += horz_sub_offset
            total_vert_offset += vert_offset
            total_horz_offset += horz_offset

    print('Saving stim-triggered plots')
    g = sns.relplot(
        data=plot_emg,
        col='label', row='elecConfig_str',
        x='time_sec', y='signal',
        hue=hue_var,
        kind='line',
        units='timestamp_usec', estimator=None,
        errorbar=None,
        height=5, aspect=2 / 5
        )

    g.set_titles(template='{row_name}\n{col_name}')
    for ax_name, this_ax in g.axes_dict.items():
        for row_idx, row in hline_df.iterrows():
            if row_idx[0] == ax_name[0]:
                this_ax.axhline(y=row['signal'], alpha=0.25, zorder=0.1)
        this_ax.axvspan(0, 100e-3, alpha=0.25, color='g', zorder=0.05)
        this_ax.axvspan(100e-3, 400e-3, alpha=0.25, color='m', zorder=0.05)
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()
