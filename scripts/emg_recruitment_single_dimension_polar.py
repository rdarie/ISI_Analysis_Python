
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
        "font.size": 10 * font_zoom_factor,
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
    'figure.titlesize': 14 * font_zoom_factor,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
sns.set(
    context='talk', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=2 * font_zoom_factor, color_codes=True, rc=snsRCParams
)
for rcK, rcV in mplRCParams.items():
    mpl.rcParams[rcK] = rcV

# folder_name = "Day12_PM"
# blocks_list = [4]
# folder_name = "Day11_PM"
# blocks_list = [2, 3]

folder_name = "Day8_AM"
# blocks_list = [1, 2, 3, 4]
blocks_list = [4]

data_path = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/3_Preprocessed_Data/{folder_name}")
pdf_folder = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/5_Figures/{folder_name}")
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)

this_emg_montage = emg_montages['lower_v2']
blocks_list_str = '_'.join(f"{block_idx}" for block_idx in blocks_list)

x_axis_name = 'freq_late'
if x_axis_name == 'freq':
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_emg_polar_recruitment_freq.pdf")
    left_sweep = 0
    right_sweep = int(0.4 * 1e6)
    amp_cutoff = 9e3
    recruitment_keys = ['elecConfig_str', 'freq']
elif x_axis_name == 'freq_late':
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_emg_polar_recruitment_freq_late.pdf")
    left_sweep = int(0.1 * 1e6)
    right_sweep = int(0.4 * 1e6)
    amp_cutoff = 9e3
    recruitment_keys = ['elecConfig_str', 'freq']
elif x_axis_name == 'amp':
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_emg_polar_recruitment_amp.pdf")
    left_sweep = 0
    right_sweep = int(0.1 * 1e6)
    freq_cutoff = 10
    recruitment_keys = ['elecConfig_str', 'amp']

verbose = 0
standardize_emg = False
normalize_across = False

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
        load_stim_info=True, force_trains=True,
        load_vicon=True, vicon_as_df=True,
        load_ripple=True, ripple_variable_names=['NEV', 'TimeCode'], ripple_as_df=True
    )
    if data_dict['vicon'] is not None:
        this_emg = data_dict['vicon']['EMG'].copy()
        this_emg.rename(columns=this_emg_montage, inplace=True)
        this_emg.drop(columns=['NA'], inplace=True)
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
n_std = 6
outlier_bounds = (auc_bar - n_std * auc_std, auc_bar + n_std * auc_std)
outlier_mask_per_trial = (auc_per_trial < outlier_bounds[0]) | (auc_per_trial > outlier_bounds[1])
outlier_mask_per_trial = outlier_mask_per_trial.any(axis='columns')
outlier_trials = outlier_mask_per_trial.index[outlier_mask_per_trial]
outlier_mask = pd.MultiIndex.from_frame(emg_metadata.loc[:, ['block', 'timestamp_usec']]).isin(outlier_trials)
#
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

#

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

average_auc_df = auc_df.groupby(recruitment_keys).mean()

delta_auc_dict = {}
for elec_a, elec_b in electrode_pairs:
    auc_a = average_auc_df.xs(elec_a, axis='index', level='elecConfig_str')
    auc_b = average_auc_df.xs(elec_b, axis='index', level='elecConfig_str')
    delta_auc_dict[elec_a] = auc_a - auc_b

delta_auc_df = pd.concat(delta_auc_dict, names=['elecConfig_str'])

determine_side = lambda x: 'Left' if x[0] == 'L' else 'Right'

auc_df.sort_index(level='elecConfig_str', key=reorder_fun, inplace=True)
average_auc_df.sort_index(level='elecConfig_str', key=reorder_fun, inplace=True)
delta_auc_df.sort_index(level='elecConfig_str', key=reorder_fun, inplace=True)


def polar_heatmapper(
        data=None,
        azimuth='label_as_degree', radius='amp', z='signal',
        label_key=[], delta_deg=None, colormesh_kwargs={},
        color=None):
    this_ax = plt.gca()
    data_square = data.pivot(index=azimuth, columns=radius, values=z)
    min_radius, max_radius = np.min(data_square.columns), np.max(data_square.columns)
    data_square.columns = data_square.columns - min_radius + 0.25 * (max_radius - min_radius)
    print(f'Radial offset: {- min_radius + 0.25 * (max_radius - min_radius)}')
    for row_idx, row in data_square.iterrows():
        row_T = row.to_frame().T
        upsampled_index = np.linspace(row_idx, row_idx + delta_deg, 10)
        sub_square = pd.concat([row_T for ii in upsampled_index], ignore_index=True)
        sub_square.index = upsampled_index
        ra, th = np.meshgrid(sub_square.columns, sub_square.index)
        this_ax.pcolormesh(th, ra, sub_square, **colormesh_kwargs)

    max_locations = data_square.abs().T.idxmax()
    this_ax.plot(max_locations.index + delta_deg / 2, max_locations.to_numpy(), 'g+')
    this_ax.plot([0, 0], [data_square.columns[0], data_square.columns[-1]], 'r-')

    this_ax.set_xticks(data_square.index + delta_deg / 2)
    this_ax.set_xticklabels(label_subset)
    return


show_plots = False
with PdfPages(pdf_path) as pdf:
    plot_auc = average_auc_df.stack().to_frame(name='signal').reset_index()
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
    elec_mask = plot_auc['elecConfig_str'].isin(elec_subset)
    label_mask = plot_auc['label'].isin(label_subset)
    plot_mask = elec_mask & label_mask

    delta_deg = 2 * np.pi / len(label_subset)
    labels_to_degrees = np.arange(delta_deg / 2, 2 * np.pi + delta_deg / 2, delta_deg)
    polar_map = {name: degree for name, degree in zip(label_subset, labels_to_degrees)}
    plot_auc.loc[:, 'label_as_degree'] = plot_auc['label'].map(polar_map)
    plot_delta_auc.loc[:, 'label_as_degree'] = plot_delta_auc['label'].map(polar_map)

    g = sns.FacetGrid(
        data=plot_auc.loc[plot_mask, :],
        col='elecConfig_str', col_wrap=6,
        # col='parent_elecConfig', row='elec_orientation',
        sharex=False,
        despine=False, subplot_kws=dict(projection='polar')
        )
    if x_axis_name in ['freq', 'freq_late']:
        this_colormap = sns.cubehelix_palette(
            start=0, rot=.4, dark=.2, light=.8,
            gamma=.75,
            as_cmap=True, reverse=True)
    elif x_axis_name == 'amp':
        this_colormap = sns.cubehelix_palette(
            start=1.5, rot=.4, dark=.2, light=.8,
            gamma=.75,
            as_cmap=True, reverse=True)

    colormesh_kws = dict(
        cmap=this_colormap,
        vmin=plot_auc.loc[plot_mask, 'signal'].min(),
        vmax=plot_auc.loc[plot_mask, 'signal'].max(),
        shading='gouraud'
        )
    if x_axis_name in ['freq', 'freq_late']:
        g.map_dataframe(
            polar_heatmapper, radius='freq',
            colormesh_kwargs=colormesh_kws,
            label_key=label_subset, delta_deg=delta_deg)
    elif x_axis_name == 'amp':
        g.map_dataframe(
            polar_heatmapper, radius='amp',
            colormesh_kwargs=colormesh_kws,
            label_key=label_subset, delta_deg=delta_deg)
    g.set_titles(template="{col_name}")
    g.figure.suptitle(f'AUC vs {x_axis_name}')
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()
    # color bar
    fig, ax = plt.subplots()
    vmin, vmax = colormesh_kws['vmin'], colormesh_kws['vmax']
    fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
            cmap=colormesh_kws['cmap']
        ),
        ax=ax
    )
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()
    #####
    #####
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
    # color bar
    fig, ax = plt.subplots()
    ##
    fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=normlize,
            cmap=colormesh_kws_delta['cmap']
        ),
        ax=ax
    )
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()
