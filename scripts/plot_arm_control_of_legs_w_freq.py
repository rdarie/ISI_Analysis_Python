
import matplotlib as mpl
import os

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

if 'CCV_HEADLESS' in os.environ:
    mpl.use('Agg')   # generate postscript output
else:
    mpl.use('QT5Agg')   # generate interactive output

import palettable
import traceback
from isicpy.utils import load_synced_mat, makeFilterCoeffsSOS
from isicpy.lookup_tables import emg_montages, kinematics_offsets, muscle_names, emg_palette, emg_hue_map
from pathlib import Path
import pandas as pd
import numpy as np
import cloudpickle as pickle
import pdb
import gc
import vg
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib import ticker
from matplotlib.lines import Line2D

useDPI = 72
dpiFactor = 72 / useDPI
font_zoom_factor = 1.

import seaborn as sns
from matplotlib import pyplot as plt

snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': 1.,
        'lines.markersize': 2.5,
        'patch.linewidth': .5,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .2,
        "font.size": 6 * font_zoom_factor,
        "axes.labelsize": 10 * font_zoom_factor,
        "axes.titlesize": 12 * font_zoom_factor,
        "xtick.labelsize": 8 * font_zoom_factor,
        "ytick.labelsize": 8 * font_zoom_factor,
        "legend.fontsize": 10 * font_zoom_factor,
        "legend.title_fontsize": 12 * font_zoom_factor,
        "xtick.bottom": True,
        "xtick.top": False,
        "ytick.left": True,
        "ytick.right": False,
        "xtick.major.width": .125,
        "ytick.major.width": .125,
        "xtick.minor.width": .125,
        "ytick.minor.width": .125,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
    }
mplRCParams = {
    'figure.titlesize': 12 * font_zoom_factor,
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

plots_dt = int(10e3)  # usec
# folder_name = "Day8_AM"
# block_idx = 4
# folder_name = "Day12_PM"
# block_idx = 4
folder_name = "Day11_AM"
block_idx = 2

data_path = Path(f"/users/rdarie/scratch/3_Preprocessed_Data/{folder_name}")
this_emg_montage = emg_montages['lower_v2']

pdf_folder = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/5_Figures/{folder_name}")
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)

pdf_path = pdf_folder / Path(f"Block_{block_idx}_arm_ctrl_legs_freq.pdf")

filterOpts = {
    'low': {
        'Wn': 500.,
        'N': 4,
        'btype': 'low',
        'ftype': 'butter'
    }
}
left_sweep = int(0 * 1e6)
right_sweep = int(20 * 1e6)
verbose = 0
standardize_emg = True

if standardize_emg:
    emg_scaler_path = data_path / "pickles" / "emg_scaler.p"
    with open(emg_scaler_path, 'rb') as handle:
        scaler = pickle.load(handle)

file_path = data_path / f"Block{block_idx:0>4d}_Synced_Session_Data.mat"
try:
    this_kin_offset = kinematics_offsets[folder_name][block_idx]
except:
    this_kin_offset = 0
data_dict = load_synced_mat(
    file_path,
    load_stim_info=True, split_trains=False,
    load_vicon=True, vicon_as_df=True, kinematics_time_offset=this_kin_offset,
    load_ripple=True, ripple_variable_names=['NEV'], ripple_as_df=True
    )

emg_df = data_dict['vicon']['EMG'].copy()
emg_df.rename(columns=this_emg_montage, inplace=True)
emg_df.index.name = 'time_usec'
emg_df.drop(['NA'], axis='columns', inplace=True)

points_df = data_dict['vicon']['Points'].copy()
# points_df.index += int(this_kin_offset)
if False:
    label_mask = points_df.columns.get_level_values('label').str.contains('ForeArm')
    for extra_label in ['Elbow', 'Foot', 'UpperArm', 'Hip', 'Knee', 'Ankle']:
        label_mask = label_mask | points_df.columns.get_level_values('label').str.contains(extra_label)
    points_df = points_df.loc[:, label_mask].copy()

angles_dict = {
    'LeftElbow': ['LeftForeArm', 'LeftElbow', 'LeftUpperArm'],
    'RightElbow': ['RightForeArm', 'RightElbow', 'RightUpperArm'],
    # 'LeftKnee': ['LeftHip', 'LeftKnee', 'LeftAnkle'],
    # 'RightKnee': ['RightHip', 'RightKnee', 'RightAnkle'],
    'LeftKnee': ['LeftLowerLeg', 'LeftKnee', 'LeftUpperLeg'],
    'RightKnee': ['RightLowerLeg', 'RightKnee', 'RightUpperLeg'],
    }
for angle_name, angle_labels in angles_dict.items():
    try:
        vec1 = (
                points_df.xs(angle_labels[0], axis='columns', level='label') -
                points_df.xs(angle_labels[1], axis='columns', level='label')
        )
        vec2 = (
                points_df.xs(angle_labels[2], axis='columns', level='label') -
                points_df.xs(angle_labels[1], axis='columns', level='label')
        )
        points_df.loc[:, (angle_name, 'angle')] = vg.angle(vec1.to_numpy(), vec2.to_numpy())
    except:
        traceback.print_exc()

lengths_dict = {
    'LeftLimb': ['LeftHip', 'LeftFoot'],
    'RightLimb': ['RightHip', 'RightFoot'],
}
for length_name, length_labels in lengths_dict.items():
    try:
        vec1 = points_df.xs(length_labels[0], axis='columns', level='label')
        vec2 = points_df.xs(length_labels[1], axis='columns', level='label')
        points_df.loc[:, (length_name, 'length')] = vg.euclidean_distance(
            vec1.to_numpy(), vec2.to_numpy())
    except:
        traceback.print_exc()

'''
points_df.columns = points_df.columns.to_frame().apply(
    lambda x: f"{x.iloc[0]}_{x.iloc[1]}", axis=1).to_list()
points_df.columns.name = 'label'
'''

stim_info_df = data_dict['stim_info']
stim_info_df.loc[:, 'elecConfig_str'] = stim_info_df.apply(
    lambda x: f'-{x["elecCath"]}+{x["elecAno"]}', axis='columns')
reversed_config_strs = stim_info_df.apply(
    lambda x: f'-{x["elecAno"]}+{x["elecCath"]}', axis='columns').unique().tolist()
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


determine_side = lambda x: 'Left' if x[0] == 'L' else 'Right'

### epoch EMG
nominal_emg_dt = np.int64(np.median(np.diff(np.asarray(emg_df.index))))
emg_downsample = int(np.ceil(plots_dt / nominal_emg_dt))
emg_sample_rate = np.round((nominal_emg_dt * 1e-6) ** -1)
emg_epoch_t = np.arange(left_sweep, right_sweep, nominal_emg_dt)
emg_nominal_num_samp = emg_epoch_t.shape[0]

nominal_kin_dt = np.int64(np.median(np.diff(np.asarray(points_df.index))))
kin_downsample = int(np.ceil(plots_dt / nominal_kin_dt))
kin_sample_rate = np.round((nominal_kin_dt * 1e-6) ** -1)
kin_epoch_t = np.arange(left_sweep, right_sweep, nominal_kin_dt)
kin_nominal_num_samp = kin_epoch_t.shape[0]

if standardize_emg:
    emg_df.loc[:, :] = scaler.transform(emg_df)
emg_df.drop(['L Forearm', 'R Forearm', 'Sync'], axis='columns', inplace=True)

filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), emg_sample_rate)
emg_df = pd.DataFrame(
    signal.sosfiltfilt(filterCoeffs, (emg_df - emg_df.mean()), axis=0),
    index=emg_df.index, columns=emg_df.columns)
# (emg_df - emg_df.mean()).abs()


# for day8_AM block 4:
# stim_info_df.index[stim_info_df.index > 135000000]
# plt.plot(stim_info_df.index, stim_info_df.index ** 0, 'o')
'''
align_timestamps = [32044466, 99050633]
vspan_limits = [
    # left
    [
        (left_sweep * 1e-6, 1.74),
        (1.74, 6.12),
        (6.12, right_sweep * 1e-6)
    ],
    # right
    [
        (left_sweep * 1e-6, 1.39),
        (1.39, 7.57),
        (7.57, right_sweep * 1e-6)
    ]
]
'''
# for day12_PM block 4:
# stim_info_df.index[stim_info_df.index > 790000000]
# stim_info_df.index[stim_info_df.index > 76000000]
# plt.plot(stim_info_df.index, stim_info_df.index ** 0, 'o')

# align_timestamps = [76021433, 797192233]
# baseline_timestamps = [int(60e6 + 8500), int(790e6 + 8500)]
# vspan_limits = [
#     # left
#     [
#         (left_sweep * 1e-6, 4.58),  # stim was already going
#         (4.58, 14.75),
#         (14.75, right_sweep * 1e-6)
#     ],
#     # right
#     [
#         (0, 4.69),  # stim starts at 0
#         (4.69, 11.96),
#         (11.96, right_sweep * 1e-6)
#     ]
# ]

# for day11_AM block 2:
# stim_info_df.index[stim_info_df.index > 370000000]
# stim_info_df.index[stim_info_df.index > xxx]
# plt.plot(stim_info_df.index, stim_info_df.index ** 0, 'o')

if True:
    align_timestamps = [348082033, 370068533]
    baseline_timestamps = [int(120e6) + 4500, int(120e6) + 4500]
    vspan_limits = [
        # left
        [
            (left_sweep * 1e-6, 3),  # stim was already going
            (3, 5),
            (5, right_sweep * 1e-6)
        ],
        # right
        [
            (left_sweep * 1e-6, 3),  # stim was already going
            (3, 5),
            (5, right_sweep * 1e-6)
        ]
    ]

    vspan_palette = palettable.colorbrewer.qualitative.Pastel2_3.mpl_colors
    vspan_colors = [
        # left
        [vspan_palette[0], vspan_palette[1], vspan_palette[0]],
        # right
        [vspan_palette[0], vspan_palette[1], vspan_palette[0]],
    ]

    all_emg_dict = {}
    all_stim_dict = {}
    all_kin_dict = {}
    metadata_fields = ['amp', 'freq', 'elecConfig_str']

    ## epoch the EMG
    for ts_idx, timestamp in enumerate(align_timestamps):
        nominal_emg_dt = np.int64(np.median(np.diff(np.asarray(points_df.index))))
        this_mask = (np.asarray(emg_df.index) >= timestamp + left_sweep) & (np.asarray(emg_df.index) <= timestamp + right_sweep)
        sweep_offset = 0
        while this_mask.sum() != emg_nominal_num_samp:
            # fix malformed epochs caused by floating point comparison errors
            if this_mask.sum() > emg_nominal_num_samp:
                sweep_offset -= nominal_emg_dt
            else:
                sweep_offset += nominal_emg_dt
            if verbose > 1:
                print(f'sweep offset set to {sweep_offset}')
            this_mask = (np.asarray(emg_df.index) >= timestamp + left_sweep - nominal_emg_dt / 2) & (np.asarray(emg_df.index) < timestamp + right_sweep + sweep_offset + nominal_emg_dt / 2)
            if verbose > 1:
                print(f'this_mask.sum() = {this_mask.sum()}')
        stim_metadata = tuple(stim_info_df.loc[timestamp, metadata_fields])
        this_entry = (timestamp,) + stim_metadata
        all_emg_dict[this_entry] = pd.DataFrame(
            emg_df.loc[this_mask, :].to_numpy(),
            index=emg_epoch_t, columns=emg_df.columns)
        all_emg_dict[this_entry].index.name = 'time_usec'

        ## epoch the stim info traces
        aligned_amp_df = data_dict['stim_info_traces']['amp'].loc[this_mask, :]
        aligned_amp_df = aligned_amp_df.loc[:, (aligned_amp_df != 0).any(axis='index')]
        aligned_freq_df = data_dict['stim_info_traces']['freq'].loc[this_mask, :]
        aligned_freq_df = aligned_freq_df.loc[:, aligned_amp_df.columns]

        this_entry = (timestamp, stim_metadata[0], stim_metadata[1],)
        all_stim_dict[this_entry] = pd.concat({'amp': aligned_amp_df, 'freq': aligned_freq_df}, names=['feature'], axis='columns')
        all_stim_dict[this_entry].index = emg_epoch_t
        all_stim_dict[this_entry].index.name = 'time_usec'
        ### epoch kinematics
        this_mask = (np.asarray(points_df.index) >= timestamp + left_sweep) & (np.asarray(points_df.index) <= timestamp + right_sweep)
        sweep_offset = 0
        while this_mask.sum() != kin_nominal_num_samp:
            # fix malformed epochs caused by floating point comparison errors
            if this_mask.sum() > kin_nominal_num_samp:
                sweep_offset -= nominal_kin_dt
            else:
                sweep_offset += nominal_kin_dt
            if verbose > 1:
                print(f'sweep offset set to {sweep_offset}')
            this_mask = (np.asarray(points_df.index) >= timestamp + left_sweep - nominal_kin_dt / 2) & (np.asarray(points_df.index) < timestamp + right_sweep + sweep_offset + nominal_kin_dt / 2)
            if verbose > 1:
                print(f'this_mask.sum() = {this_mask.sum()}')

        this_entry = (timestamp,) + stim_metadata
        all_kin_dict[this_entry] = pd.DataFrame(
            points_df.loc[this_mask, :].to_numpy(),
            index=kin_epoch_t, columns=points_df.columns)
        all_kin_dict[this_entry].index.name = 'time_usec'
        # baseline based on defined timestamp
        all_kin_dict[this_entry] = all_kin_dict[this_entry] - points_df.loc[baseline_timestamps[ts_idx], :]
        # baseline based on first entry
        # all_kin_dict[this_entry] = all_kin_dict[this_entry] - all_kin_dict[this_entry].iloc[0, :]

    aligned_emg_df = pd.concat(all_emg_dict, names=['timestamp_usec'] + metadata_fields)
    aligned_stim_info = pd.concat(all_stim_dict, names=['timestamp_usec', 'amp', 'freq'])
    aligned_kin_df = pd.concat(all_kin_dict, names=['timestamp_usec'] + metadata_fields)

    for cn in data_dict['stim_info_traces']['amp'].columns:
        print(f'{cn}\n\tAccording to traces, unique amps are:')
        print(f"\t{data_dict['stim_info_traces']['amp'][cn].unique()}")
        if cn in stim_info_df['elecConfig_str'].tolist():
            print(f'\tAccording to stim_info, unique amps are:')
            print(f"\t{stim_info_df.loc[stim_info_df['elecConfig_str'] == cn, 'amp'].unique()}")

    displacements_list = []
    columns_list = []
    for name, group in aligned_kin_df.groupby('label', axis='columns'):
        try:
            displacements_list.append(np.sqrt(
                group.xs('x', axis='columns', level='axis') ** 2 +
                group.xs('y', axis='columns', level='axis') ** 2 +
                group.xs('z', axis='columns', level='axis') ** 2
                ))
            columns_list.append((name, 'displacement'))
        except:
            traceback.print_exc()

    disps_df = pd.concat(displacements_list, axis='columns')
    disps_df.columns = pd.MultiIndex.from_tuples(columns_list, names=['label', 'axis'])
    aligned_kin_df = pd.concat([aligned_kin_df, disps_df], axis='columns')

    pretty_points_label_lookup = {
        'LeftKnee': 'Left knee', 'RightKnee': 'Right knee',
        'LeftElbow': 'Left elbow', 'RightElbow': 'Right elbow',
        'LeftAnkle': 'Left ankle', 'RightAnkle': 'Right ankle',
        'RightFoot': 'Right foot', 'LeftFoot': 'Left foot',
        'LeftLowerLeg': 'Left shin', 'RightLowerLeg': 'Right shin',
        'LeftUpperLeg': 'Left thigh', 'RightUpperLeg': 'Right thigh',
    }
    prettify_points_label = lambda x: f'{pretty_points_label_lookup.get(x["label"])} ({x["axis"]})'
    prettify_points_label_tuple = lambda x: f'{pretty_points_label_lookup.get(x[0])} ({x[1]})'

    kin_format_df = pd.read_csv('./kin_format_info.csv')
    kin_format_df.loc[:, 'color'] = [
        palettable.colorbrewer.qualitative.Set3_12.mpl_colors[idx]
        for idx in kin_format_df['palette_idx']]
    # kin_palette = [
    #     palettable.colorbrewer.qualitative.Paired_12.mpl_colors[idx]
    #     for idx in [8, 10, 9, 11]
    #     # for idx in [5, 5, 5, 8, 6, 6, 6, 9]
    # ]


    vspan_alpha = 0.1
    elec_format_df = pd.read_csv('./elec_format_info.csv')
    elec_format_df.loc[:, 'which_array'] = elec_format_df.loc[:, 'which_array'].str.replace(' ', '')
    elec_format_df.loc[:, 'pretty_label'] = elec_format_df.loc[:, 'pretty_label'].str.replace(' ', '')
    stim_palette = [
        palettable.colorbrewer.qualitative.Paired_12.mpl_colors[idx]
        for idx in elec_format_df['palette_idx']
    ]

    notable_electrode_labels = elec_format_df['label'].to_list()

    electrode_functional_names = {
        rw['label']: rw['pretty_label'] for rw_idx, rw in elec_format_df.iterrows()
    }

    base_electrode_hue_map = {
        nm: c for nm, c in zip(notable_electrode_labels, stim_palette)
    }
    def elec_reorder_fun(config_strings):
        return pd.Index([notable_electrode_labels.index(name) for name in config_strings], name=config_strings.name)

    # aligned_stim_info.sort_index(axis='columns', level='elecConfig_str', key=reorder_fun, inplace=True)

    show_plots = True
    with PdfPages(pdf_path) as pdf:
        ###
        # emg_label_subset = ['LVL', 'LMH', 'LTA', 'LMG', 'LSOL', 'RLVL', 'RMH', 'RTA', 'RMG', 'RSOL']
        emg_label_subset = ['LVL', 'LMH', 'LMG', 'RLVL', 'RMH', 'RMG']
        # plot_emg['label'].unique().tolist() , 'LSOL' , 'RMH' , 'RSOL'
        plot_emg = (aligned_emg_df.loc[:, emg_label_subset].iloc[::emg_downsample, :]).stack().to_frame(name='signal').reset_index()
        plot_emg.loc[:, 'time_sec'] = plot_emg['time_usec'] * 1e-6
        ##
        plot_stim_traces = aligned_stim_info.xs('amp', level='feature', drop_level=False, axis='columns').iloc[::emg_downsample, :].stack(level=['elecConfig_str', 'feature']).to_frame(name='signal').reset_index()
        plot_stim_traces.sort_values(by='elecConfig_str', key=elec_reorder_fun, inplace=True)
        plot_stim_traces.loc[:, 'time_sec'] = plot_stim_traces['time_usec'] * 1e-6
        plot_stim_traces.loc[:, 'signal'] = plot_stim_traces['signal'] * 1e-3  # convert to mA
        ## offset rostral:
        caudal_amp_offset = 20
        amp_ticks = [0, 10, caudal_amp_offset, caudal_amp_offset + 10]
        amp_ticklabels = ['0', '10', '0', '10']
        plot_stim_traces.loc[:, 'which_array'] = plot_stim_traces['elecConfig_str'].map(elec_format_df.set_index('label')['which_array'])
        plot_stim_traces.loc[plot_stim_traces['which_array'].isin(['caudal']), 'signal'] += caudal_amp_offset
        ##
        plot_stim_freq = aligned_stim_info.xs('freq', level='feature', drop_level=False, axis='columns').iloc[::emg_downsample, :].stack(level=['elecConfig_str', 'feature']).to_frame(name='signal').reset_index()
        plot_stim_freq.sort_values(by='elecConfig_str', key=elec_reorder_fun, inplace=True)
        plot_stim_freq.loc[:, 'time_sec'] = plot_stim_freq['time_usec'] * 1e-6
        plot_stim_freq.loc[:, 'signal'] = plot_stim_freq['signal']
        ## offset rostral:
        plot_stim_freq.loc[:, 'which_array'] = plot_stim_freq['elecConfig_str'].map(elec_format_df.set_index('label')['which_array'])
        caudal_freq_offset = 125
        freq_ticks = [0, 100, caudal_freq_offset, caudal_freq_offset + 100]
        freq_ticklabels = ['0', '100', '0', '100']
        plot_stim_freq.loc[plot_stim_freq['which_array'].isin(['caudal']), 'signal'] += caudal_freq_offset
        ## plot_stim_freq
        ##
        # points_label_subset = [('LeftLimb', 'length'), ('RightLimb', 'length'), ('LeftElbow', 'angle'), ('RightElbow', 'angle')]
        control_label_subset = [
            # ('LeftElbow', 'angle'), ('RightElbow', 'angle'),
            ('LeftKnee', 'angle'), ('RightKnee', 'angle'),
            ]
        outcome_label_subset = [
            # ('LeftKnee', 'angle'), ('RightKnee', 'angle'),
            ('LeftAnkle', 'displacement'), ('RightAnkle', 'displacement'),
            # ('LeftAnkle', 'x'), ('RightAnkle', 'x'),
            # ('LeftAnkle', 'y'), ('RightAnkle', 'y'),
            # ('LeftAnkle', 'z'), ('RightAnkle', 'z'),
        ]
        which_outcome = 'angle'
        points_label_subset = control_label_subset + outcome_label_subset

        plot_kin = aligned_kin_df.loc[:, points_label_subset].iloc[::kin_downsample, :].stack(level=['label', 'axis']).to_frame(name='signal').reset_index()
        plot_kin.loc[:, 'time_sec'] = plot_kin['time_usec'] * 1e-6
        plot_kin.loc[:, 'pretty_label'] = plot_kin.apply(prettify_points_label, axis='columns')
        pretty_points_labels = [pretty_points_label_lookup[lbl] for lbl in plot_kin['label'].unique()]
        pretty_outcome_labels = [prettify_points_label_tuple(lbl) for lbl in outcome_label_subset]
        pretty_control_labels = [prettify_points_label_tuple(lbl) for lbl in control_label_subset]
        fig = plt.figure(figsize=(7, 7))
        gs = fig.add_gridspec(
            5, 3, height_ratios=(1, 1, 1, 2, 1), width_ratios=(8, 8, 1),
            left=0.025, right=0.975, bottom=0.025, top=0.975,
            wspace=0.025, hspace=0.025
        )
        # Create the Axes.
        kin_ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        kin_ax[0].get_shared_y_axes().join(kin_ax[0], kin_ax[1])
        stim_ax = [fig.add_subplot(gs[1, 0], sharex=kin_ax[0]), fig.add_subplot(gs[1, 1], sharex=kin_ax[1])]
        stim_ax[0].get_shared_y_axes().join(stim_ax[0], stim_ax[1])
        freq_ax = [fig.add_subplot(gs[2, 0], sharex=kin_ax[0]), fig.add_subplot(gs[2, 1], sharex=kin_ax[1])]
        freq_ax[0].get_shared_y_axes().join(freq_ax[0], freq_ax[1])
        emg_ax = [fig.add_subplot(gs[3, 0], sharex=kin_ax[0]), fig.add_subplot(gs[3, 1], sharex=kin_ax[1])]
        emg_ax[0].get_shared_y_axes().join(emg_ax[0], emg_ax[1])
        outcome_ax = [fig.add_subplot(gs[4, 0], sharex=kin_ax[0]), fig.add_subplot(gs[4, 1], sharex=kin_ax[1])]
        outcome_ax[0].get_shared_y_axes().join(outcome_ax[0], outcome_ax[1])

        this_emg_palette = [emg_hue_map[nm] for nm in emg_label_subset]
        this_emg_hue_map = {
            nm: emg_hue_map[nm] for nm in emg_label_subset
            }

        vert_span = plot_emg['signal'].max() - plot_emg['signal'].min()
        vert_offset = -30e-2 * vert_span
        horz_span = plot_emg['time_sec'].max() - plot_emg['time_sec'].min()
        horz_offset = 0 * horz_span
        n_offsets = 0

        vert_offset_lookup = {}
        for name in emg_label_subset:
            group_index = plot_emg.loc[plot_emg['label'] == name, :].index
            plot_emg.loc[group_index, 'signal'] = plot_emg.loc[group_index, 'signal'] + n_offsets * vert_offset
            plot_emg.loc[group_index, 'time_sec'] = plot_emg.loc[group_index, 'time_sec'] + n_offsets * horz_offset
            vert_offset_lookup[name] = n_offsets * vert_offset
            n_offsets += 1

        left_mask_emg = plot_emg['timestamp_usec'] == align_timestamps[0]
        right_mask_emg = plot_emg['timestamp_usec'] == align_timestamps[1]
        left_mask_stim = plot_stim_traces['timestamp_usec'] == align_timestamps[0]
        right_mask_stim = plot_stim_traces['timestamp_usec'] == align_timestamps[1]
        '''
        for name, c in emg_hue_map.items():
            for this_ax in emg_ax:
                this_ax.axhline(y=vert_offset_lookup[name], c=c)
                '''
        sns.lineplot(
            data=plot_emg.loc[left_mask_emg, :], ax=emg_ax[0],
            x='time_sec', y='signal',
            hue='label', palette=this_emg_hue_map, legend=False)
        sns.lineplot(
            data=plot_emg.loc[right_mask_emg, :], ax=emg_ax[1],
            x='time_sec', y='signal',
            hue='label', palette=this_emg_hue_map, legend=False)

        dummy_legend_handle = mpl.patches.Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor='none', visible=False)
        custom_legend_lines = [dummy_legend_handle] + [
            Line2D([0], [0], color=c, lw=2)
            for idx, c in enumerate(this_emg_palette)
        ]

        determine_side = lambda x: 'Left  ' if x[0] == 'L' else 'Right'
        pretty_emg_labels = [f'         {muscle_names[n]}' for n in emg_label_subset]
        pretty_emg_labels[0] = f'{determine_side(emg_label_subset[0])} {muscle_names[emg_label_subset[0]]}'
        pretty_emg_labels[3] = f'{determine_side(emg_label_subset[3])} {muscle_names[emg_label_subset[3]]}'

        time_ticks = [0, 10, 20]
        for this_ax in emg_ax:
            this_ax.xaxis.set_major_locator(ticker.FixedLocator(time_ticks))
            # this_ax.xaxis.set_major_formatter()
            this_ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(left_sweep * 1e-6, right_sweep * 1e-6, 1)))
            # this_ax.xaxis.set_minor_formatter()
            this_ax.set_xlim((left_sweep * 1e-6, right_sweep * 1e-6))

        emg_ax[0].set_ylabel('Normalized\nEMG (a.u.)')
        emg_ax[0].set_yticklabels([])
        emg_ax[1].legend(custom_legend_lines, ['EMG'] + pretty_emg_labels, loc='center left', bbox_to_anchor=(1.025, .5), borderaxespad=0.)
        emg_ax[1].set_yticklabels([])
        emg_ax[1].set_ylabel('')

        for this_ax in emg_ax:
            plt.setp(this_ax.get_xticklabels(), visible=False)

        # reorder to match left-right order
        electrode_labels = elec_format_df.loc[elec_format_df['label'].isin(plot_stim_traces['elecConfig_str']), :].sort_values(['which_array', 'palette_idx'])['label']
        electrode_hue_map = {
            nm: base_electrode_hue_map[nm] for nm in electrode_labels
            }
        # pretty_electrode_labels = [f'{electrode_functional_names[cfg]} ({frequencies_each_config[cfg]} Hz)' for cfg in electrode_labels]
        pretty_electrode_labels = [f'{electrode_functional_names[cfg]}' for cfg in electrode_labels]

        sns.lineplot(
            data=plot_stim_traces.loc[left_mask_stim, :], ax=stim_ax[0],
            x='time_sec', y='signal',
            hue='elecConfig_str', style='feature',
            palette=electrode_hue_map, legend=False)
        sns.lineplot(
            data=plot_stim_traces.loc[right_mask_stim, :], ax=stim_ax[1],
            x='time_sec', y='signal',
            hue='elecConfig_str', style='feature',
            palette=electrode_hue_map, legend=False)
        ##
        sns.lineplot(
            data=plot_stim_freq.loc[left_mask_stim, :], ax=freq_ax[0],
            x='time_sec', y='signal',
            hue='elecConfig_str', style='feature',
            palette=electrode_hue_map, legend=False)
        sns.lineplot(
            data=plot_stim_freq.loc[right_mask_stim, :], ax=freq_ax[1],
            x='time_sec', y='signal',
            hue='elecConfig_str', style='feature',
            palette=electrode_hue_map, legend=False)

        all_custom_stim_legend_lines = [
            Line2D([0], [0], color=electrode_hue_map[nm], lw=2)
            for idx, nm in enumerate(electrode_labels)
        ]
        custom_legend_lines = (
                [dummy_legend_handle, dummy_legend_handle] +
                all_custom_stim_legend_lines[:4] + [dummy_legend_handle] +
                all_custom_stim_legend_lines[4:])

        stim_ax[1].legend(
            custom_legend_lines,
            ['Stim. electrode\nconfiguration', 'Caudal'] + pretty_electrode_labels[:4] + ['Rostral'] + pretty_electrode_labels[4:],
            loc='center left', bbox_to_anchor=(1.025, .5), borderaxespad=0.)

        for this_ax in stim_ax:
            plt.setp(this_ax.get_xticklabels(), visible=False)
            this_ax.yaxis.set_major_locator(ticker.FixedLocator(amp_ticks))

        for this_ax in freq_ax:
            plt.setp(this_ax.get_xticklabels(), visible=False)
            this_ax.yaxis.set_major_locator(ticker.FixedLocator(freq_ticks))

        stim_ax[1].set_yticklabels([])
        stim_ax[1].set_ylabel('')
        stim_ax[0].set_ylabel('Stim.\namplitude\n(mA)')
        stim_ax[0].set_yticklabels(amp_ticklabels)
        ##
        freq_ax[1].set_yticklabels([])
        freq_ax[1].set_ylabel('')
        freq_ax[0].set_ylabel('Stim.\nfrequency\n(Hz)')
        freq_ax[0].set_yticklabels(freq_ticklabels)

        kinematics_labels = plot_kin['pretty_label'].unique().tolist()
        kin_hue_map = {
            nm: kin_format_df.set_index('pretty_label').loc[nm, 'color']
            for nm in kinematics_labels
        }
        ###############

        left_mask_kin = (plot_kin['timestamp_usec'] == align_timestamps[0]) & (plot_kin['pretty_label'].isin(pretty_control_labels))
        right_mask_kin = (plot_kin['timestamp_usec'] == align_timestamps[1]) & (plot_kin['pretty_label'].isin(pretty_control_labels))
        sns.lineplot(
            data=plot_kin.loc[left_mask_kin, :], ax=kin_ax[0],
            x='time_sec', y='signal',
            # hue='pretty_label', lw=2,
            hue='pretty_label', palette=kin_hue_map, legend=False
        )
        sns.lineplot(
            data=plot_kin.loc[right_mask_kin, :], ax=kin_ax[1],
            x='time_sec', y='signal',
            # hue='pretty_label', lw=2,
            hue='pretty_label', palette=kin_hue_map, legend=False
        )

        present_points = plot_kin.loc[plot_kin['pretty_label'].isin(pretty_control_labels), 'pretty_label'].unique()
        custom_legend_lines = [dummy_legend_handle] + [
            Line2D([0], [0], color=kin_hue_map[lbl], lw=2)
            for idx, lbl in enumerate(present_points)
        ]
        kin_ax[1].legend(
            custom_legend_lines, ['Joint'] + present_points.tolist(),
            loc='center left', bbox_to_anchor=(1.025, .5), borderaxespad=0.)

        for this_ax in kin_ax:
            this_ax.xaxis.set_major_locator(ticker.FixedLocator(time_ticks))
            this_ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(left_sweep * 1e-6, right_sweep * 1e-6, 1)))
            this_ax.yaxis.set_major_locator(ticker.FixedLocator([-45, 0, 45]))
            plt.setp(this_ax.get_xticklabels(), visible=False)

        kin_ax[1].set_yticklabels([])
        kin_ax[1].set_ylabel('')
        kin_ax[0].set_ylabel('Joint angle\n(deg.)')
        #######################

        left_mask_kin = (plot_kin['timestamp_usec'] == align_timestamps[0]) & (plot_kin['pretty_label'].isin(pretty_outcome_labels))
        right_mask_kin = (plot_kin['timestamp_usec'] == align_timestamps[1]) & (plot_kin['pretty_label'].isin(pretty_outcome_labels))
        sns.lineplot(
            data=plot_kin.loc[left_mask_kin, :], ax=outcome_ax[0],
            x='time_sec', y='signal',
            # hue='pretty_label', lw=2,
            hue='pretty_label', palette=kin_hue_map, legend=False
        )
        sns.lineplot(
            data=plot_kin.loc[right_mask_kin, :], ax=outcome_ax[1],
            x='time_sec', y='signal',
            hue='pretty_label', palette=kin_hue_map, legend=False
        )

        present_points = plot_kin.loc[plot_kin['pretty_label'].isin(pretty_outcome_labels), 'pretty_label'].unique()
        custom_legend_lines = [dummy_legend_handle] + [
            Line2D([0], [0], color=kin_hue_map[lbl], lw=2)
            for idx, lbl in enumerate(present_points)
        ]
        outcome_ax[1].legend(
            custom_legend_lines, ['Joint'] + present_points.tolist(),
            loc='center left', bbox_to_anchor=(1.025, .5), borderaxespad=0.)

        for this_ax in outcome_ax:
            this_ax.xaxis.set_major_locator(ticker.FixedLocator(time_ticks))
            this_ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(left_sweep * 1e-6, right_sweep * 1e-6, 1)))
            this_ax.set_xlabel('Time (sec.)')
            # if which_outcome == 'angle':
            #     this_ax.yaxis.set_major_locator(ticker.FixedLocator([-45, 0, 45]))
            # plt.setp(this_ax.get_xticklabels(), visible=False)

        outcome_ax[1].set_yticklabels([])
        outcome_ax[1].set_ylabel('')
        if which_outcome == 'displacement':
            outcome_ax[0].set_ylabel('Marker displacement\n(mm)')
        elif which_outcome == 'angle':
            outcome_ax[0].set_ylabel('joint angle\n(deg.)')

        for lr_idx, this_emg_ax in enumerate(emg_ax):
            these_lims = vspan_limits[lr_idx]
            for train_idx, lims in enumerate(these_lims):
                this_emg_ax.axvspan(*lims, facecolor=vspan_colors[lr_idx][train_idx], alpha=vspan_alpha)
                kin_ax[lr_idx].axvspan(*lims, facecolor=vspan_colors[lr_idx][train_idx], alpha=vspan_alpha)
                stim_ax[lr_idx].axvspan(*lims, facecolor=vspan_colors[lr_idx][train_idx], alpha=vspan_alpha)
                freq_ax[lr_idx].axvspan(*lims, facecolor=vspan_colors[lr_idx][train_idx], alpha=vspan_alpha)
                outcome_ax[lr_idx].axvspan(*lims, facecolor=vspan_colors[lr_idx][train_idx], alpha=vspan_alpha)

        fig.align_labels()
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if show_plots:
            plt.show()
        else:
            plt.close()
