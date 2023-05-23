
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
useDPI = 144
dpiFactor = 72 / useDPI
font_zoom_factor = 1.

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': .5,
        'lines.markersize': 2.,
        'patch.linewidth': .5,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .25,
        "grid.linewidth": .2,
        "font.size": 5 * font_zoom_factor,
        "axes.axisbelow": False,
        "axes.labelpad": -1 * font_zoom_factor,
        "axes.labelsize": 5 * font_zoom_factor,
        "axes.titlesize": 6 * font_zoom_factor,
        "xtick.labelsize": 5 * font_zoom_factor,
        "ytick.labelsize": 5 * font_zoom_factor,
        "legend.fontsize": 5 * font_zoom_factor,
        "legend.title_fontsize": 6 * font_zoom_factor,
        "xtick.bottom": True,
        "xtick.top": False,
        "ytick.left": True,
        "ytick.right": False,
        "xtick.major.width": .25,
        "ytick.major.width": .25,
        "xtick.minor.width": .25,
        "ytick.minor.width": .25,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
    }
mplRCParams = {
    'figure.titlesize': 6 * font_zoom_factor,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
sns.set(
    context='paper', style='white',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True, rc=snsRCParams
)
for rcK, rcV in mplRCParams.items():
    mpl.rcParams[rcK] = rcV

import matplotlib as mpl
import os

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

if 'CCV_HEADLESS' in os.environ:
    mpl.use('Agg')   # generate postscript output
else:
    mpl.use('QT5Agg')   # generate interactive output

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

# folder_name = "Day2_AM"
# blocks_list = [3]
# this_emg_montage = emg_montages['lower']

# folder_name = "Day12_PM"
# blocks_list = [4]

folder_name = "Day11_PM"
blocks_list = [2, 3]
this_emg_montage = emg_montages['lower_v2']

# folder_name = "Day8_AM"
# blocks_list = [1, 2, 3, 4]
# blocks_list = [4]

# folder_name = "Day1_AM"
# blocks_list = [2, 3]

# folder_name = "Day12_AM"
# blocks_list = [2, 3]

data_path = Path(f"/users/rdarie/scratch/3_Preprocessed_Data/{folder_name}")
pdf_folder = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/5_Figures/{folder_name}")
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)

blocks_list_str = '_'.join(f"{block_idx}" for block_idx in blocks_list)

x_axis_name = 'freq_late'
if x_axis_name == 'freq':
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_emg_spiderweb_recruitment_freq.pdf")
    data_export_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_emg_spiderweb_recruitment_freq_data.parquet")
    left_sweep = 0
    right_sweep = int(0.3 * 1e6)
    amp_cutoff = 11e3
    recruitment_keys = ['elecConfig_str', 'freq']
    max_marker_color = 'm'
elif x_axis_name == 'freq_late':
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_emg_spiderweb_recruitment_freq_late.pdf")
    data_export_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_emg_spiderweb_recruitment_freq_late_data.parquet")
    left_sweep = int(0.1 * 1e6)
    right_sweep = int(0.3 * 1e6)
    amp_cutoff = 11e3
    recruitment_keys = ['elecConfig_str', 'freq']
    max_marker_color = 'm'
elif x_axis_name == 'amp':
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_emg_spiderweb_recruitment_amp.pdf")
    data_export_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_emg_spiderweb_recruitment_amp_data.parquet")
    left_sweep = 0
    right_sweep = int(0.1 * 1e6)
    freq_cutoff = 10
    recruitment_keys = ['elecConfig_str', 'amp']
    max_marker_color = 'g'

verbose = 2
standardize_emg = True
normalize_across = False

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
            load_stim_info=True, split_trains=True, stim_info_traces=False, force_trains=True,
            load_vicon=True, vicon_as_df=True,
            load_ripple=True, ripple_variable_names=['NEV'], ripple_as_df=True,
            )
        if data_dict['vicon'] is not None:
            this_emg = data_dict['vicon']['EMG'].copy()
            this_emg.rename(columns=this_emg_montage, inplace=True)
            this_emg.drop(columns=['NA'], inplace=True)
        if data_dict['stim_info'] is not None:
            all_stim_info[block_idx] = data_dict['stim_info']

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
    epoch_t = np.arange(left_sweep, right_sweep, nominal_dt)
    nominal_num_samp = epoch_t.shape[0]

    if standardize_emg:
        emg_scaler_path = data_path / "pickles" / "emg_scaler.p"
        with open(emg_scaler_path, 'rb') as handle:
            scaler = pickle.load(handle)
        this_emg.loc[:, :] = scaler.transform(this_emg)

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
            this_mask = (
                    (analog_time_vector >= timestamp + left_sweep - nominal_dt / 2) &
                    (analog_time_vector < timestamp + right_sweep + sweep_offset + nominal_dt / 2))
            if verbose > 1:
                print(f'this_mask.sum() = {this_mask.sum()}')
        aligned_dfs[timestamp] = pd.DataFrame(
            this_emg.loc[this_mask, :].to_numpy(), index=epoch_t, columns=this_emg.columns)
    all_aligned_emg[block_idx] = pd.concat(aligned_dfs, names=['timestamp_usec', 'time_usec'])

stim_info_df = pd.concat(all_stim_info, names=['block', 'timestamp_usec'])
stim_info_df.loc[:, 'elecConfig_str'] = stim_info_df.apply(lambda x: f'-{x["elecCath"]}+{x["elecAno"]}', axis='columns')
# stim_info_df.groupby('elecConfig_str').count()
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
emg_df.drop(['L Forearm', 'R Forearm', 'Sync'], axis='columns', inplace=True)

'''
del aligned_dfs, all_aligned_emg, all_stim_info
gc.collect()
g = sns.displot(data=stim_info_df, x='delta_timestamp_usec', rug=True, element='step', fill=False)
plt.show()
'''
emg_metadata = emg_df.index.to_frame()

for meta_key in ['elecConfig_str', 'freq', 'amp']:
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

if x_axis_name in ['freq', 'freq_late']:
    # remove amp <= cutoff
    stim_info_df = stim_info_df.loc[stim_info_df['amp'] >= amp_cutoff, :]
    emg_df = emg_df.loc[emg_df.index.get_level_values('amp') >= amp_cutoff, :]
elif x_axis_name == 'amp':
    # remove freq >= cutoff
    stim_info_df = stim_info_df.loc[stim_info_df['freq'] <= freq_cutoff, :]
    emg_df = emg_df.loc[emg_df.index.get_level_values('freq') <= freq_cutoff, :]
##########################
## export data for git repo
emg_df.to_parquet(data_export_path)
##########################
auc_df = emg_df.groupby(recruitment_keys + ['block', 'timestamp_usec']).mean()
temp_average_auc = auc_df.groupby(recruitment_keys).mean()

if normalize_across:
    scaler = MinMaxScaler()
    scaler.fit(auc_df.stack().to_frame())
    auc_df = auc_df.apply(lambda x: scaler.transform(x.reshape(-1, 1)).flatten(), raw=True, axis='index')
else:
    scaler = MinMaxScaler()
    # scaler.fit(auc_df)
    scaler.fit(temp_average_auc)
    auc_df.loc[:, :] = scaler.transform(auc_df)

# auc_df.reset_index().groupby(recruitment_keys).count().max()
average_auc_df = auc_df.groupby(recruitment_keys).mean()

should_plot_delta_auc = False
if should_plot_delta_auc:
    delta_auc_dict = {}
    for elec_a, elec_b in electrode_pairs:
        auc_a = average_auc_df.xs(elec_a, axis='index', level='elecConfig_str')
        auc_b = average_auc_df.xs(elec_b, axis='index', level='elecConfig_str')
        delta_auc_dict[elec_a] = auc_a - auc_b
    delta_auc_df = pd.concat(delta_auc_dict, names=['elecConfig_str'])

    auc_df.sort_index(level='elecConfig_str', key=reorder_fun, inplace=True)
    average_auc_df.sort_index(level='elecConfig_str', key=reorder_fun, inplace=True)

    delta_auc_df.sort_index(level='elecConfig_str', key=reorder_fun, inplace=True)


determine_side = lambda x: 'Left' if x[0] == 'L' else 'Right'
def polar_webmapper(
        data=None, ax=None, ax_var='elecConfig_str',
        azimuth='label_as_degree', radius='signal', hue='amp',
        color_map=None, color_norm=None,
        marker_at_max=True, show_titles=False, delta_deg=None):
    plot_data = data.sort_values(azimuth)
    if marker_at_max:
        for this_deg, deg_group in plot_data.groupby(azimuth):
            hue_of_max = deg_group.set_index(hue)[radius].idxmax()
            ax.plot(
                this_deg + delta_deg / 2, deg_group[radius].max(), '+',
                color=color_map(color_norm(hue_of_max)), zorder=6, )
    new_x_ticks = None
    for hue_name, hue_group in plot_data.groupby(hue):
        plot_vec = hue_group.set_index(azimuth)[radius]
        az = (plot_vec.index + delta_deg / 2).to_list()
        r = plot_vec.to_list()
        if new_x_ticks is None:
            new_x_ticks = az.copy()
            new_x_labels = hue_group['label'].to_list()
        ## wrap around
        az.append(az[0])
        r.append(r[0])
        ax.plot(az, r, color=color_map(color_norm(hue_name)))

    ax.set_xticks(new_x_ticks)
    ax.set_xticklabels(new_x_labels)
    ax.set_yticks([])
    # ax.set_yticklabels([])
    ax.tick_params(pad=snsRCParams['axes.labelpad'])
    if show_titles:
        ax.set_title(data[ax_var].unique()[0])
    return

dummy_legend_handle = mpl.patches.Rectangle(
    (0, 0), 1, 1, fill=False, edgecolor='none', visible=False)

show_plots = True
with PdfPages(pdf_path) as pdf:
    fig = plt.figure(figsize=(7.15, 3))
    gs = mpl.gridspec.GridSpec(
        nrows=6, ncols=8,
        width_ratios=[2] * 3 + [1.5] + [2] * 3 + [1.5],
        figure=fig, wspace=0.25, hspace=0.35,
        left=0, right=1, top=1, bottom=0,
        )
    legend_ax = fig.add_subplot(gs[2, 7])
    extra_legend_ax = fig.add_subplot(gs[5, 6])
    axes_dict = {}
    axes_dict['-[3]+[2]'] = fig.add_subplot(gs[1:3, 0], projection='polar')
    axes_dict['-[2]+[3]'] = fig.add_subplot(gs[3:5, 0], projection='polar')
    #
    axes_dict['-[3]+[10]'] = fig.add_subplot(gs[0:2, 1], projection='polar')
    axes_dict['-[2]+[10]'] = fig.add_subplot(gs[4:6, 1], projection='polar')
    #
    axes_dict['-[10]+[3]'] = fig.add_subplot(gs[1:3, 2], projection='polar')
    axes_dict['-[10]+[2]'] = fig.add_subplot(gs[3:5, 2], projection='polar')
    #
    axes_dict['-[18]+[27]'] = fig.add_subplot(gs[1:3, 4], projection='polar')
    axes_dict['-[18]+[26]'] = fig.add_subplot(gs[3:5, 4], projection='polar')
    #
    axes_dict['-[27]+[18]'] = fig.add_subplot(gs[0:2, 5], projection='polar')
    axes_dict['-[26]+[18]'] = fig.add_subplot(gs[4:6, 5], projection='polar')
    #
    axes_dict['-[27]+[26]'] = fig.add_subplot(gs[1:3, 6], projection='polar')
    axes_dict['-[26]+[27]'] = fig.add_subplot(gs[3:5, 6], projection='polar')

    plot_auc = average_auc_df.stack().to_frame(name='signal').reset_index()
    if should_plot_delta_auc:
        plot_delta_auc = delta_auc_df.stack().to_frame(name='signal').reset_index()
    ##
    plot_auc.loc[:, 'side'] = plot_auc['label'].apply(determine_side)
    plot_auc.loc[:, 'muscle'] = plot_auc['label'].map(muscle_names)
    plot_auc.loc[:, 'parent_elecConfig'] = plot_auc['elecConfig_str'].map(parent_elec_configurations)
    plot_auc.loc[:, 'elec_orientation'] = plot_auc['elecConfig_str'].map(orientation_types)
    ###
    elec_subset = plot_auc['elecConfig_str'].unique().tolist()  #  ['-(2,)+(3,)', '-(3,)+(2,)',]
    # label_subset = ['LVL', 'LMH', 'LTA', 'LMG', 'LSOL', 'RLVL', 'RMH', 'RTA', 'RMG', 'RSOL']  #  plot_auc['label'].unique().tolist()
    label_subset = ['RLVL', 'RMH', 'LMH', 'LVL', 'LTA', 'LMG', 'LSOL', 'RSOL', 'RMG', 'RTA', ]  # ordered for polar plot
    ###
    if x_axis_name == 'amp':
        plot_auc.loc[:, 'amp_mA'] = plot_auc['amp'] / 1e3
    elec_mask = plot_auc['elecConfig_str'].isin(elec_subset)
    label_mask = plot_auc['label'].isin(label_subset)
    plot_mask = elec_mask & label_mask
    delta_deg = 2 * np.pi / len(label_subset)
    labels_to_degrees = np.arange(delta_deg / 2, 2 * np.pi + delta_deg / 2, delta_deg)
    polar_map = {name: degree for name, degree in zip(label_subset, labels_to_degrees)}
    plot_auc.loc[:, 'label_as_degree'] = plot_auc['label'].map(polar_map)
    if should_plot_delta_auc:
        plot_delta_auc.loc[:, 'label_as_degree'] = plot_delta_auc['label'].map(polar_map)

    if x_axis_name in ['freq', 'freq_late']:
        hue_var = 'freq'
        this_colormap = sns.cubehelix_palette(
            start=0, rot=.4, dark=.2, light=.8,
            gamma=.75, as_cmap=True, reverse=True)
        x_axis_pretty_name = 'Stim.\nfreq. (Hz)'
        this_color_norm = plt.Normalize(
            vmin=plot_auc.loc[plot_mask, 'freq'].min() - 1e-6,
            vmax=plot_auc.loc[plot_mask, 'freq'].max() + 1e-6)
        legend_hues = [7, 25, 50, 75, 100]
        legend_hues_normed = [this_color_norm(xx) for xx in legend_hues]
        custom_legend_text = [x_axis_pretty_name] + [
            f'{xx:.3g}' for xx in legend_hues
        ]
        custom_legend_lines = [dummy_legend_handle] + [
            Line2D([0], [0], color=this_colormap(xx), lw=1)
            for xx in legend_hues_normed
        ]
        max_legend_line = Line2D([0], [0], color='m', markerfacecolor='m', lw=0, marker='+', markersize=3,)
        max_legend_text = 'Freq. of max.\nrecruitment (at 11 mA)'
    elif x_axis_name == 'amp':
        hue_var = 'amp_mA'
        this_colormap = sns.cubehelix_palette(
            start=1.5, rot=.4, dark=.2, light=.8,
            gamma=.75, as_cmap=True, reverse=True)
        this_color_norm = plt.Normalize(
            vmin=plot_auc.loc[plot_mask, 'amp_mA'].min() - 1e-6,
            vmax=plot_auc.loc[plot_mask, 'amp_mA'].max() + 1e-6)
        x_axis_pretty_name = 'Stim.\namp. (mA)'
        num_legend_lines = 5
        legend_hues_normed = np.linspace(0, 1, num_legend_lines)
        legend_hues = [this_color_norm.inverse(xx) for xx in legend_hues_normed]
        custom_legend_text = [x_axis_pretty_name] + [
            f'{xx:.2g}' for xx in legend_hues
        ]
        custom_legend_lines = [dummy_legend_handle] + [
            Line2D([0], [0], color=this_colormap(xx), lw=1)
            for xx in legend_hues_normed
        ]
        max_legend_line = Line2D([0], [0], color='g', markerfacecolor='g', lw=0, markersize=3, marker='+')
        max_legend_text = 'Amp. of max.\nrecruitment (at 7-10 Hz)'

    for name, group in plot_auc.groupby('elecConfig_str'):
        if name in axes_dict:
            polar_webmapper(
                data=group, ax=axes_dict[name],
                azimuth='label_as_degree', radius='signal', hue=hue_var,
                color_map=this_colormap, color_norm=this_color_norm,
                marker_at_max=True, delta_deg=delta_deg)

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    sns.despine(ax=legend_ax, left=True, bottom=True)
    leg = legend_ax.legend(
        custom_legend_lines, custom_legend_text,
        loc='center left', bbox_to_anchor=(0, 0), borderaxespad=0.)
    extra_legend_ax.set_xticks([])
    extra_legend_ax.set_yticks([])
    sns.despine(ax=extra_legend_ax, left=True, bottom=True)
    extra_leg = extra_legend_ax.legend(
        [max_legend_line], [max_legend_text],
        loc='center left', bbox_to_anchor=(0, .5), borderaxespad=0.)

    fig.align_labels()
    pdf.savefig(
        bbox_inches='tight', pad_inches=0,
        bbox_extra_artists=[leg, extra_leg])
    if show_plots:
        plt.show()
    else:
        plt.close()
    #####
    #####
    if should_plot_delta_auc:
        elec_mask_delta = plot_delta_auc['elecConfig_str'].isin(elec_subset)
        label_mask_delta = plot_delta_auc['label'].isin(label_subset)
        plot_mask_delta = elec_mask_delta & label_mask_delta
        g = sns.FacetGrid(
            data=plot_delta_auc,
            col='elecConfig_str', col_wrap=6,
            # col='parent_elecConfig', row='elec_orientation',
            despine=False, subplot_kws=dict(projection='polar')
            )
        vmin = plot_delta_auc.loc[plot_mask_delta, 'signal'].min()
        vmax = plot_delta_auc.loc[plot_mask_delta, 'signal'].max()
        center = 0
        vrange = max(vmax - center, center - vmin)
        normlize = mpl.colors.Normalize(center - vrange, center + vrange)
        this_colormap = sns.color_palette('vlag', as_cmap=True)
        colormesh_kws_delta = dict(
            cmap=this_colormap,
            norm=normlize,
            shading='gouraud'
        )
        print('Saving delta heatmap')
        if x_axis_name in ['freq', 'freq_late']:
            g.map_dataframe(
                polar_heatmapper, radius='freq',
                colormesh_kwargs=colormesh_kws_delta,
                label_key=label_subset, delta_deg=delta_deg)
        elif x_axis_name == 'amp':
            g.map_dataframe(
                polar_heatmapper, radius='amp',
                colormesh_kwargs=colormesh_kws_delta,
                label_key=label_subset, delta_deg=delta_deg)
        g.figure.suptitle(f'delta AUC vs {x_axis_name}')
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if show_plots:
            plt.show()
        else:
            plt.close()
