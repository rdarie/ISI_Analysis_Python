
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
from isicpy.utils import load_synced_mat, makeFilterCoeffsSOS, timestring_to_timestamp
from isicpy.lookup_tables import emg_montages, kinematics_offsets, muscle_names, emg_palette, emg_hue_map, video_info
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
import matplotlib.path as mpath
from matplotlib import ticker
from matplotlib.lines import Line2D

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
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
        "axes.axisbelow": False,
        "axes.labelsize": 9 * font_zoom_factor,
        "axes.titlesize": 11 * font_zoom_factor,
        "xtick.labelsize": 8 * font_zoom_factor,
        "ytick.labelsize": 8 * font_zoom_factor,
        "legend.fontsize": 9 * font_zoom_factor,
        "legend.title_fontsize": 11 * font_zoom_factor,
        "xtick.bottom": True,
        "xtick.top": False,
        "ytick.left": True,
        "ytick.right": False,
        "xtick.major.width": .5,
        "ytick.major.width": .5,
        "xtick.minor.width": .25,
        "ytick.minor.width": .25,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
    }
mplRCParams = {
    'figure.titlesize': 11 * font_zoom_factor,
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

plots_dt = int(2e3)  # usec
# folder_name = "Day8_AM"
# block_idx = 4
# folder_name = "Day12_PM"
# block_idx = 4
folder_name = "Day11_AM"
block_idx = 4

angles_dict = {
    'LeftElbow': ['LeftForeArm', 'LeftElbow', 'LeftUpperArm'],
    'RightElbow': ['RightForeArm', 'RightElbow', 'RightUpperArm'],
    # 'LeftKnee': ['LeftHip', 'LeftKnee', 'LeftAnkle'],
    # 'RightKnee': ['RightHip', 'RightKnee', 'RightAnkle'],
    'LeftKnee': ['LeftLowerLeg', 'LeftKnee', 'LeftUpperLeg'],
    'RightKnee': ['RightLowerLeg', 'RightKnee', 'RightUpperLeg'],
    }

data_path = Path(f"/users/rdarie/scratch/3_Preprocessed_Data/{folder_name}")
this_emg_montage = emg_montages['lower_v2']

pdf_folder = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/5_Figures/{folder_name}")
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)
pdf_path = pdf_folder / Path(f"Block_{block_idx}_arm_ctrl_legs_freq_rasters.pdf")

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
    load_ripple=True, ripple_variable_names=['NEV', 'TimeCode'], ripple_as_df=True
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
stim_info_df.loc[:, 'elecAll'] = stim_info_df.apply(lambda x: x['elecCath'] + x['elecAno'], axis='columns')
nev_spikes = data_dict['ripple']['NEV'].copy()

## how I figured out the conversion
# nev_elecs = np.unique(nev_spikes['Electrode'])
# si_elecs = np.unique(stim_info_df.loc[:, 'elecAll'].sum())

conversion = lambda x:  8 - (x - 1) % 8 + 8 * ((x - 1) // 8)
nev_spikes.loc[:, 'Electrode'] = nev_spikes['Electrode'].apply(conversion)

nev_spikes.loc[:, 'amp'] = np.nan
nev_spikes.loc[:, 'freq'] = np.nan

for ii, (this_timestamp, group) in enumerate(stim_info_df.groupby('timestamp_usec')):
    for row_idx, row in group.iterrows():
        if ii == 0:
            cath_mask = nev_spikes['Electrode'].isin(row['elecCath'])
            ano_mask = nev_spikes['Electrode'].isin(row['elecAno'])
        elif ii != (stim_info_df.groupby('timestamp_usec').ngroups - 1):
            cath_mask = nev_spikes['Electrode'].isin(row['elecCath']) & (nev_spikes['time_usec'] >= this_timestamp)
            ano_mask = nev_spikes['Electrode'].isin(row['elecAno']) & (nev_spikes['time_usec'] >= this_timestamp)
        nev_spikes.loc[ano_mask, 'amp'] = row['amp']
        nev_spikes.loc[cath_mask, 'amp'] = row['amp'] * (-1)
        nev_spikes.loc[ano_mask | cath_mask, 'freq'] = row['freq']
        nev_spikes.loc[ano_mask | cath_mask, 'elecConfig_str'] = row['elecConfig_str']
# for day8_AM block 4:
# stim_info_df.index[stim_info_df.index > 135000000]
# plt.plot(stim_info_df.index, stim_info_df.index ** 0, 'o')
'''
has_audible_timing = False
legend_halfway_idx = 2
# emg_label_subset = ['LVL', 'LMH', 'LMG', 'RLVL', 'RMH', 'RMG']
control_label_subset = [
    ('LeftElbow', 'angle'), ('RightElbow', 'angle'),
]
outcome_label_subset = [
    ('LeftKnee', 'angle'), ('RightKnee', 'angle'),
]
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
'''
has_audible_timing = False
legend_halfway_idx = 2
emg_label_subset = ['LVL', 'LMH', 'LMG', 'RLVL', 'RMH', 'RMG']
only_these_electrodes = ["-(14,)+(6, 22)", "-(3,)+(2,)", "-(139,)+(131,)", "-(136,)+(144,)"]
align_timestamps = [76021433, 797192233]
which_outcome = 'angle'
control_label_subset = [
    ('LeftElbow', 'angle'), ('RightElbow', 'angle'),
]
outcome_label_subset = [
    ('LeftKnee', 'angle'), ('RightKnee', 'angle'),
]
which_outcome = 'angle'
color_background = True
baseline_timestamps = [int(60e6 + 8500), int(790e6 + 8500)]
vspan_limits = [
    # left
    [
        (left_sweep * 1e-6, 4.58),  # stim was already going
        (4.58, 14.75),
        (14.75, right_sweep * 1e-6)
    ],
    # right
    [
        (0, 4.69),  # stim starts at 0
        (4.69, 11.96),
        (11.96, right_sweep * 1e-6)
    ]
]
'''
# for day11_AM block 4:
# stim_info_df.index[stim_info_df.index > 480010033]
# stim_info_df.index[stim_info_df.index > xxx]
# plt.plot(stim_info_df.index, stim_info_df.index ** 0, 'o')
align_timestamps = [500046533, 480117533]
baseline_timestamps = [int(181e6) + 1500, int(181e6) + 1500]

# for day11_AM block 2:
# stim_info_df.index[stim_info_df.index > 370000000]
# stim_info_df.index[stim_info_df.index > xxx]
# plt.plot(stim_info_df.index, stim_info_df.index ** 0, 'o')
# align_timestamps = [348082033, 370068533]
# baseline_timestamps = [int(120e6) + 4500, int(120e6) + 4500]

has_audible_timing = True
legend_halfway_idx = 4
emg_label_subset = ['LVL', 'LMH', 'LTA', 'LMG', 'RLVL', 'RMH', 'RTA', 'RMG']
control_label_subset = [
    ('LeftKnee', 'angle'), ('RightKnee', 'angle'),
]
outcome_label_subset = [
    ('LeftToe', 'z'), ('RightToe', 'z'),
]
which_outcome = 'displacement'
color_background = False

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

def int_or_nan(x):
    try:
        return int(x)
    except:
        return np.nan

fps = 29.97
audible_timing_path = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/6_Video/Day11_AM_Cam1_GH010630_audible_timings.txt")
audible_timing = pd.read_csv(audible_timing_path)
audible_timing = audible_timing.stack().reset_index().iloc[:, 1:]
audible_timing.columns = ['words', 'time']
audible_timing.loc[:, ['m', 's', 'f']] = audible_timing['time'].apply(lambda x: x.split(':')).apply(pd.Series).to_numpy()
audible_timing.loc[:, ['m', 's', 'f']] = audible_timing.loc[:, ['m', 's', 'f']].applymap(int_or_nan)
audible_timing.loc[:, 'total_frames'] = audible_timing['m'] * 60 * 30 + audible_timing['s'] * 30 + audible_timing['f']
# video_t = audible_timing.loc[:, 'total_frames'].apply(lambda x: pd.Timedelta(x / 29.97, unit='sec'))
# audible_timing.loc[:, 'timestamp'] = pd.Timestamp(year=2022, month=10, day=31) + video_t

first_ripple = data_dict['ripple']['TimeCode'].iloc[0, :]
ripple_origin_timestamp = timestring_to_timestamp(first_ripple['TimeString'], day=31, fps=fps, timecode_type='NDF')
audible_timing.loc[:, 'timedeltas'] = audible_timing.loc[:, 'total_frames'].apply(lambda x: pd.Timedelta(x / 29.97, unit='sec'))
audible_timecodes = pd.Timestamp(year=2022, month=10, day=31) + audible_timing.loc[:, 'timedeltas']
audible_timing.loc[:, 'ripple_time'] = (audible_timecodes - ripple_origin_timestamp).apply(lambda x: x.total_seconds()) + first_ripple['PacketTime']
audible_timing.dropna(inplace=True)
audible_timing.loc[:, 'time_usec'] = audible_timing['ripple_time'].apply(lambda x: int(x * 1e6))

vspan_palette = palettable.colorbrewer.qualitative.Pastel2_3.mpl_colors
vspan_colors = [
    # left
    [vspan_palette[0], vspan_palette[1], vspan_palette[0]],
    # right
    [vspan_palette[0], vspan_palette[1], vspan_palette[0]],
]

all_emg_dict = {}
all_kin_dict = {}
all_spikes_dict = {}
if has_audible_timing:
    all_audible_dict = {}
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
    # no baseline
    #
    # baseline based on defined timestamp
    all_kin_dict[this_entry] = all_kin_dict[this_entry] - points_df.loc[baseline_timestamps[ts_idx], :]
    # baseline based on first entry
    # all_kin_dict[this_entry] = all_kin_dict[this_entry] - all_kin_dict[this_entry].iloc[0, :]
    this_mask = (nev_spikes['time_usec'] >= (timestamp + int(left_sweep))) & (nev_spikes['time_usec'] <= (timestamp + int(right_sweep)))
    all_spikes_dict[timestamp] = nev_spikes.loc[this_mask, :].copy()
    all_spikes_dict[timestamp].loc[:, 'time_usec'] = all_spikes_dict[timestamp]['time_usec'] - timestamp
    if has_audible_timing:
        this_mask = (audible_timing['time_usec'] >= (timestamp + int(left_sweep))) & (audible_timing['time_usec'] <= (timestamp + int(right_sweep)))
        all_audible_dict[timestamp] = audible_timing.loc[this_mask, :].copy()
        all_audible_dict[timestamp].loc[:, 'time_usec'] = all_audible_dict[timestamp]['time_usec'] - timestamp


aligned_emg_df = pd.concat(all_emg_dict, names=['timestamp_usec'] + metadata_fields)

aligned_kin_df = pd.concat(all_kin_dict, names=['timestamp_usec'] + metadata_fields)
aligned_spikes_df = pd.concat(all_spikes_dict, names=['timestamp_usec'])
if has_audible_timing:
    aligned_audible_df = pd.concat(all_audible_dict, names=['timestamp_usec'])

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
    'LeftToe': 'Left toe', 'RightToe': 'Right toe',
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
# elec_format_df.loc[:, 'pretty_label'] = elec_format_df.loc[:, 'pretty_label'].str.replace(' ', '')
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


class GradientLine:
    def __init__(self, cmap=None):
        self.cmap = cmap.copy()


class GradientLineHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        legend_x = np.linspace(x0, x0 + width, 100)
        legend_y = np.zeros((100,)) + y0 + height / 2
        # needs to be (numlines) x (points per line) x 2 (for x and y)
        points = np.array([legend_x, legend_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        legend_line = mpl.collections.LineCollection(segments, cmap=orig_handle.cmap)
        legend_line.set_array(np.linspace(0, 1, 100))
        legend_line.set_linewidth(3)
        handlebox.add_artist(legend_line)
        return legend_line

show_plots = True
with PdfPages(pdf_path) as pdf:
    ###
    if True:
        # emg_label_subset = ['LVL', 'LMH', 'LTA', 'LMG', 'LSOL', 'RLVL', 'RMH', 'RTA', 'RMG', 'RSOL']
        #
        # plot_emg['label'].unique().tolist()
        plot_emg = (aligned_emg_df.loc[:, emg_label_subset].iloc[::emg_downsample, :]).stack().to_frame(name='signal').reset_index()
        plot_emg.loc[:, 'time_sec'] = plot_emg['time_usec'] * 1e-6
        plot_stim_spikes = aligned_spikes_df.reset_index()
        if has_audible_timing:
            plot_audible = aligned_audible_df.reset_index()
        points_label_subset = control_label_subset + outcome_label_subset
        try:
            plot_stim_spikes = plot_stim_spikes.loc[plot_stim_spikes['elecConfig_str'].isin(only_these_electrodes), :]
        except:
            pass

        plot_kin = aligned_kin_df.loc[:, points_label_subset].iloc[::kin_downsample, :].stack(level=['label', 'axis']).to_frame(name='signal').reset_index()
        plot_kin.loc[:, 'time_sec'] = plot_kin['time_usec'] * 1e-6
        plot_kin.loc[:, 'pretty_label'] = plot_kin.apply(prettify_points_label, axis='columns')
        pretty_points_labels = [pretty_points_label_lookup[lbl] for lbl in plot_kin['label'].unique()]
        pretty_outcome_labels = [prettify_points_label_tuple(lbl) for lbl in outcome_label_subset]
        pretty_control_labels = [prettify_points_label_tuple(lbl) for lbl in control_label_subset]
        fig = plt.figure(figsize=(8, 8))
        if has_audible_timing:
            gs = fig.add_gridspec(
                # 5, 3, height_ratios=(1, 1, 1, 2, 1), width_ratios=(8, 8, 1),
                5, 3, height_ratios=(2, 2, 5, 2, 1), width_ratios=(8, 8, 1),
                left=0.025, right=0.975, bottom=0.025, top=0.975,
                wspace=0.025, hspace=0.025
            )
        else:
            gs = fig.add_gridspec(
                # 5, 3, height_ratios=(1, 1, 1, 2, 1), width_ratios=(8, 8, 1),
                4, 3, height_ratios=(2, 1, 5, 2), width_ratios=(8, 8, 1),
                left=0.025, right=0.975, bottom=0.025, top=0.975,
                wspace=0.025, hspace=0.025
            )
        # Create the Axes.
        kin_ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        kin_ax[0].get_shared_y_axes().join(kin_ax[0], kin_ax[1])
        stim_ax = [fig.add_subplot(gs[1, 0], sharex=kin_ax[0]), fig.add_subplot(gs[1, 1], sharex=kin_ax[1])]
        stim_ax[0].get_shared_y_axes().join(stim_ax[0], stim_ax[1])
        # freq_ax = [fig.add_subplot(gs[2, 0], sharex=kin_ax[0]), fig.add_subplot(gs[2, 1], sharex=kin_ax[1])]
        # freq_ax[0].get_shared_y_axes().join(freq_ax[0], freq_ax[1])
        emg_ax = [fig.add_subplot(gs[2, 0], sharex=kin_ax[0]), fig.add_subplot(gs[2, 1], sharex=kin_ax[1])]
        emg_ax[0].get_shared_y_axes().join(emg_ax[0], emg_ax[1])
        outcome_ax = [fig.add_subplot(gs[3, 0], sharex=kin_ax[0]), fig.add_subplot(gs[3, 1], sharex=kin_ax[1])]
        outcome_ax[0].get_shared_y_axes().join(outcome_ax[0], outcome_ax[1])

        if has_audible_timing:
            verbal_ax = [fig.add_subplot(gs[4, 0], sharex=kin_ax[0]), fig.add_subplot(gs[4, 1], sharex=kin_ax[1])]
            verbal_ax[0].get_shared_y_axes().join(verbal_ax[0], verbal_ax[1])
            bottom_ax = verbal_ax
        else:
            bottom_ax = outcome_ax


        this_emg_palette = [emg_hue_map[nm] for nm in emg_label_subset]
        this_emg_hue_map = {
            nm: emg_hue_map[nm] for nm in emg_label_subset
            }

        vert_span = plot_emg['signal'].max() - plot_emg['signal'].min()
        vert_offset = -30e-2 * vert_span
        horz_span = plot_emg['time_sec'].max() - plot_emg['time_sec'].min()
        horz_offset = 0 * horz_span
        n_offsets = 0

        emg_major_yticks = []
        emg_minor_yticks = []
        emg_yticklabels = []
        vert_offset_lookup = {}
        for name in emg_label_subset:
            group_index = plot_emg.loc[plot_emg['label'] == name, :].index
            plot_emg.loc[group_index, 'signal'] = plot_emg.loc[group_index, 'signal'] + n_offsets * vert_offset
            plot_emg.loc[group_index, 'time_sec'] = plot_emg.loc[group_index, 'time_sec'] + n_offsets * horz_offset
            vert_offset_lookup[name] = n_offsets * vert_offset
            emg_major_yticks += [n_offsets * vert_offset]
            emg_minor_yticks += [(n_offsets - 0.25) * vert_offset, (n_offsets + 0.25) * vert_offset]
            emg_yticklabels += [f"{-0.25 * vert_offset:.2f}", "0.", f"{0.25 * vert_offset:.2f}"]
            n_offsets += 1

        left_mask_emg = plot_emg['timestamp_usec'] == align_timestamps[0]
        right_mask_emg = plot_emg['timestamp_usec'] == align_timestamps[1]

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
            Line2D([0], [0], color=c, lw=3)
            for idx, c in enumerate(this_emg_palette)
        ]

        determine_side = lambda x: 'Left  ' if x[0] == 'L' else 'Right'
        pretty_emg_labels = [f'         {muscle_names[n]}' for n in emg_label_subset]
        idx_of_half = int(len(emg_label_subset) / 2)
        pretty_emg_labels[0] = f'{determine_side(emg_label_subset[0])} {muscle_names[emg_label_subset[0]]}'
        pretty_emg_labels[idx_of_half] = f'{determine_side(emg_label_subset[idx_of_half])} {muscle_names[emg_label_subset[idx_of_half]]}'

        time_ticks = [0, 10, 20]
        for this_ax in emg_ax:
            this_ax.xaxis.set_major_locator(ticker.FixedLocator(time_ticks))
            # this_ax.xaxis.set_major_formatter()
            this_ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(left_sweep * 1e-6, right_sweep * 1e-6, 1)))
            # this_ax.xaxis.set_minor_formatter()
            this_ax.set_xlim((left_sweep * 1e-6, right_sweep * 1e-6))
            plt.setp(this_ax.get_xticklabels(), visible=False)
            #
            # this_ax.yaxis.set_major_locator(ticker.FixedLocator(sorted(emg_major_yticks)))
            # this_ax.yaxis.set_minor_locator(ticker.FixedLocator(sorted(emg_minor_yticks)))
            this_ax.yaxis.set_major_locator(ticker.MultipleLocator(np.abs(vert_offset)))
            this_ax.yaxis.set_minor_locator(ticker.MultipleLocator(np.abs(vert_offset) / 2))
            plt.setp(this_ax.get_yticklabels(), visible=False)

        emg_ax[0].set_ylabel('Normalized\nEMG (a.u.)')
        # emg_ax[0].set_yticklabels(emg_yticklabels)
        emg_ax[1].legend(custom_legend_lines, ['EMG'] + pretty_emg_labels, loc='center left', bbox_to_anchor=(1.025, .5), borderaxespad=0.)
        emg_ax[1].set_ylabel('')

        # reorder to match left-right order
        electrode_labels = elec_format_df.loc[elec_format_df['label'].isin(plot_stim_spikes['elecConfig_str']), :].sort_values(['which_array', 'palette_idx'])['label']
        electrode_hue_map = {
            nm: base_electrode_hue_map[nm] for nm in electrode_labels
            }
        pretty_electrode_labels = [f'{electrode_functional_names[cfg]}' for cfg in electrode_labels]

    if has_audible_timing:
        audible_palette = palettable.colorbrewer.qualitative.Set3_12.mpl_colors
        audible_hue_map = {
            'Left Foot': audible_palette[5],
            'Right Foot': audible_palette[6]
            }
        # plot_audible.loc[:, 'adjusted_time'] = plot_audible['time_usec'] * 1e-6
        # plot_audible.loc[:, ['time', 'adjusted_time']]
        left_mask_audible = (plot_audible['timestamp_usec'] == align_timestamps[0])
        x = plot_audible.loc[left_mask_audible, 'time_usec'].drop_duplicates() * 1e-6
        origins = np.concatenate([x.to_numpy().reshape(1, -1), np.zeros((1, x.shape[0])) - 0.1], axis=0).T
        endpoints = origins.copy()
        endpoints[:, 1] -= .8
        segments = np.concatenate([origins[:, np.newaxis, :], endpoints[:, np.newaxis, :]], axis=1)
        lc = mpl.collections.LineCollection(segments, colors=plot_audible.loc[left_mask_audible, 'words'].map(audible_hue_map).to_list())
        lc.set_linewidth(3)
        line = verbal_ax[0].add_collection(lc)
        right_mask_audible = (plot_audible['timestamp_usec'] == align_timestamps[1])
        x = plot_audible.loc[right_mask_audible, 'time_usec'].drop_duplicates() * 1e-6
        origins = np.concatenate([x.to_numpy().reshape(1, -1), np.zeros((1, x.shape[0])) - 0.1], axis=0).T
        endpoints = origins.copy()
        endpoints[:, 1] -= .8
        segments = np.concatenate([origins[:, np.newaxis, :], endpoints[:, np.newaxis, :]], axis=1)
        lc = mpl.collections.LineCollection(segments, colors=plot_audible.loc[right_mask_audible, 'words'].map(audible_hue_map).to_list())
        lc.set_linewidth(3)
        line = verbal_ax[1].add_collection(lc)

    for this_ax in verbal_ax:
        this_ax.set_xlim([left_sweep * 1e-6, right_sweep * 1e-6])
        this_ax.set_ylim([-1., 0.])

    verbal_legend_lines = [dummy_legend_handle] + [
        Line2D([0], [0], color=value, lw=2)
        for key, value in audible_hue_map.items()
    ]
    verbal_ax[1].legend(
        verbal_legend_lines, ["Report", '"Left"', '"Right"'],
        loc='center left', bbox_to_anchor=(1.025, .5), borderaxespad=0.)

    raster_linewidth = 1.
    mixin_color = np.asarray([1., 1., 1.])
    max_mixin = 0.5
    skip_every = 10
    all_custom_stim_legend_lines = []

    for line_idx, config_str in enumerate(electrode_labels):
        left_mask_spikes = (plot_stim_spikes['timestamp_usec'] == align_timestamps[0]) & (plot_stim_spikes['elecConfig_str'] == config_str)
        right_mask_spikes = (plot_stim_spikes['timestamp_usec'] == align_timestamps[1]) & (plot_stim_spikes['elecConfig_str'] == config_str)

        base_color = np.asarray(electrode_hue_map[config_str])
        cmap = LinearSegmentedColormap.from_list(
            name='custom', colors=[
                base_color,
                base_color + (mixin_color - base_color) * max_mixin])
        norm = plt.Normalize(
            vmin=plot_stim_spikes.loc[left_mask_spikes | right_mask_spikes, 'amp'].abs().min() - 1e-6,
            vmax=plot_stim_spikes.loc[left_mask_spikes | right_mask_spikes, 'amp'].max() + 1e-6)
        #
        x = plot_stim_spikes.loc[left_mask_spikes, 'time_usec'].drop_duplicates() * 1e-6
        z = plot_stim_spikes.loc[x.index, 'amp'].abs().to_numpy()[::skip_every]
        origins = np.concatenate([x.to_numpy().reshape(1, -1), np.zeros((1, x.shape[0])) - line_idx - 0.1], axis=0).T
        origins = origins[::skip_every, :]
        endpoints = origins.copy()
        endpoints[:, 1] -= .8
        segments = np.concatenate([origins[:, np.newaxis, :], endpoints[:, np.newaxis, :]], axis=1)
        lc = mpl.collections.LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(z)
        lc.set_linewidth(raster_linewidth)
        line = stim_ax[0].add_collection(lc)
        #
        x = plot_stim_spikes.loc[right_mask_spikes, 'time_usec'].drop_duplicates() * 1e-6
        z = plot_stim_spikes.loc[x.index, 'amp'].abs().to_numpy()[::skip_every]
        origins = np.concatenate([x.to_numpy().reshape(1, -1), np.zeros((1, x.shape[0])) - line_idx - .1], axis=0).T
        origins = origins[::skip_every, :]
        endpoints = origins.copy()
        endpoints[:, 1] -= .8
        segments = np.concatenate([origins[:, np.newaxis, :], endpoints[:, np.newaxis, :]], axis=1)
        lc = mpl.collections.LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(z)
        lc.set_linewidth(raster_linewidth)
        line = stim_ax[1].add_collection(lc)
        #
        all_custom_stim_legend_lines.append(GradientLine(cmap))

    stim_ax[0].set_xlim([left_sweep * 1e-6, right_sweep * 1e-6])
    stim_ax[0].set_ylim([- line_idx - 1.5, .5])
    stim_ax[1].set_xlim([left_sweep * 1e-6, right_sweep * 1e-6])
    stim_ax[1].set_ylim([- line_idx - 1.5, .5])

    '''all_custom_stim_legend_lines = [
        Line2D([0], [0], color=electrode_hue_map[nm], lw=2)
        for idx, nm in enumerate(electrode_labels)
    ]'''
    custom_legend_lines = (
            [dummy_legend_handle] * 2 +
            all_custom_stim_legend_lines[:legend_halfway_idx] + [dummy_legend_handle] +
            all_custom_stim_legend_lines[legend_halfway_idx:])
    custom_legend_texts = (
            ['Stim. electrode\nconfiguration', 'Caudal'] +
            pretty_electrode_labels[:legend_halfway_idx] +
            ['Rostral'] + pretty_electrode_labels[legend_halfway_idx:])

    stim_ax[1].legend(
        custom_legend_lines, custom_legend_texts,
        handler_map={GradientLine: GradientLineHandler()},
        loc='center left', bbox_to_anchor=(1.025, .5), borderaxespad=0.)

    for this_ax in stim_ax:
        plt.setp(this_ax.get_xticklabels(), visible=False)
        plt.setp(this_ax.get_yticklabels(), visible=False)
        this_ax.set_yticks([])

    stim_ax[1].set_ylabel('')
    stim_ax[0].set_ylabel('Stim. pulses')
    ##
    '''
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

    for this_ax in freq_ax:
        plt.setp(this_ax.get_xticklabels(), visible=False)
        this_ax.yaxis.set_major_locator(ticker.FixedLocator(freq_ticks))

    ##
    freq_ax[1].set_yticklabels([])
    freq_ax[1].set_ylabel('')
    freq_ax[0].set_ylabel('Stim.\nfrequency\n(Hz)')
    freq_ax[0].set_yticklabels(freq_ticklabels)
        '''
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
        Line2D([0], [0], color=kin_hue_map[lbl], lw=3)
        for idx, lbl in enumerate(present_points)
    ]
    kin_ax[1].legend(
        custom_legend_lines, ['Joint'] + present_points.tolist(),
        loc='center left', bbox_to_anchor=(1.025, .5), borderaxespad=0.)

    for this_ax in kin_ax:
        this_ax.xaxis.set_major_locator(ticker.FixedLocator(time_ticks))
        this_ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(left_sweep * 1e-6, right_sweep * 1e-6, 1)))
        # this_ax.yaxis.set_major_locator(ticker.FixedLocator([-45, 0, 45]))
        this_ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
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
        # hue='pretty_label', lw=2,
        hue='pretty_label', palette=kin_hue_map, legend=False
    )

    present_points = plot_kin.loc[plot_kin['pretty_label'].isin(pretty_outcome_labels), 'pretty_label'].unique()
    custom_legend_lines = [dummy_legend_handle] + [
        Line2D([0], [0], color=kin_hue_map[lbl], lw=3)
        for idx, lbl in enumerate(present_points)
    ]
    outcome_ax[1].legend(
        custom_legend_lines, ['Joint'] + present_points.tolist(),
        loc='center left', bbox_to_anchor=(1.025, .5), borderaxespad=0.)

    for this_ax in outcome_ax:
        this_ax.xaxis.set_major_locator(ticker.FixedLocator(time_ticks))
        this_ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(left_sweep * 1e-6, right_sweep * 1e-6, 1)))
        this_ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        # if which_outcome == 'angle':
        #     this_ax.yaxis.set_major_locator(ticker.FixedLocator([-45, 0, 45]))
        plt.setp(this_ax.get_xticklabels(), visible=False)

    outcome_ax[1].set_yticklabels([])
    outcome_ax[1].set_ylabel('')
    if which_outcome == 'displacement':
        outcome_ax[0].set_ylabel('Marker height\n(mm)')
    elif which_outcome == 'angle':
        outcome_ax[0].set_ylabel('joint angle\n(deg.)')

    if has_audible_timing:
        verbal_ax[0].set_ylabel('Verbal report\nof stim')
        for this_ax in verbal_ax:
            this_ax.xaxis.set_major_locator(ticker.FixedLocator(time_ticks))
            this_ax.set_yticklabels([])

    for this_ax in bottom_ax:
        this_ax.set_xlabel('Time (sec.)')
        plt.setp(this_ax.get_xticklabels(), visible=True)

    if color_background:
        for lr_idx, this_emg_ax in enumerate(emg_ax):
            these_lims = vspan_limits[lr_idx]
            for train_idx, lims in enumerate(these_lims):
                this_emg_ax.axvspan(*lims, facecolor=vspan_colors[lr_idx][train_idx], alpha=vspan_alpha)
                kin_ax[lr_idx].axvspan(*lims, facecolor=vspan_colors[lr_idx][train_idx], alpha=vspan_alpha)
                stim_ax[lr_idx].axvspan(*lims, facecolor=vspan_colors[lr_idx][train_idx], alpha=vspan_alpha)
                # freq_ax[lr_idx].axvspan(*lims, facecolor=vspan_colors[lr_idx][train_idx], alpha=vspan_alpha)
                outcome_ax[lr_idx].axvspan(*lims, facecolor=vspan_colors[lr_idx][train_idx], alpha=vspan_alpha)
                if has_audible_timing:
                    verbal_ax[lr_idx].axvspan(*lims, facecolor=vspan_colors[lr_idx][train_idx], alpha=vspan_alpha)

    fig.align_labels()
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()
