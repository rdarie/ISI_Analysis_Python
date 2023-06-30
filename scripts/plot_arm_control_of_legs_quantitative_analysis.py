
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
from isicpy.utils import makeFilterCoeffsSOS, closestSeries
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
import matplotlib.transforms as transforms
from matplotlib import ticker
from matplotlib.lines import Line2D

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
useDPI = 144
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
        "axes.linewidth": .25,
        "grid.linewidth": .2,
        "font.size": 5 * font_zoom_factor,
        "axes.axisbelow": False,
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
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1,
        "ytick.minor.size": 1,
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

plots_dt = int(1e3)  #  usec

# folder_name = "Day7_AM"
# block_idx = 4
# this_emg_montage = emg_montages['lower']

folder_name = "Day8_AM"
block_idx = 2
this_emg_montage = emg_montages['lower']

# folder_name = "Day11_AM"
# block_idx = 4
# this_emg_montage = emg_montages['lower_v2']

# folder_name = "Day12_PM"
# block_idx = 4
# this_emg_montage = emg_montages['lower_v2']

if folder_name in ['Day11_AM']:
    angles_dict = {
        'LeftElbow': ['LeftForeArm', 'LeftElbow', 'LeftUpperArm'],
        'RightElbow': ['RightForeArm', 'RightElbow', 'RightUpperArm'],
        'LeftKnee': ['LeftLowerLeg', 'LeftKnee', 'LeftUpperLeg'],
        'RightKnee': ['RightLowerLeg', 'RightKnee', 'RightUpperLeg'],
        }
elif folder_name in ['Day8_AM', 'Day12_PM']:
    # angles_dict = {
    #     'LeftElbow': ['LeftForeArm', 'LeftElbow', 'LeftUpperArm'],
    #     'RightElbow': ['RightForeArm', 'RightElbow', 'RightUpperArm'],
    #     'LeftKnee': ['LeftHip', 'LeftKnee', 'LeftAnkle'],
    #     'RightKnee': ['RightHip', 'RightKnee', 'RightAnkle'],
    #     }
    angles_dict = {
        'LeftElbow': ['LeftUpperArm', 'LeftForeArm', 'LeftElbow', ],
        'RightElbow': ['RightUpperArm', 'RightForeArm', 'RightElbow', ],
        'LeftKnee': ['LeftAnkle', 'LeftHip', 'LeftKnee', ],
        'RightKnee': ['RightAnkle', 'RightHip', 'RightKnee', ],
        }

data_path = Path(f"/users/rdarie/Desktop/ISI-C-003/3_Preprocessed_Data/{folder_name}")
pdf_folder = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/5_Figures/{folder_name}")

if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)
pdf_path = pdf_folder / Path(f"{folder_name}_Block_{block_idx}_arm_ctrl_legs_freq_rasters.pdf")

filterOptsEnvelope = {
    'low': {
        'Wn': 10.,
        'N': 8,
        'btype': 'low',
        'ftype': 'butter'
    }
}
filterOpts = {
    'low': {
        'Wn': 500.,
        'N': 4,
        'btype': 'low',
        'ftype': 'butter'
    }
}

left_sweep = int(0 * 1e6)
right_sweep = int(10 * 1e6)
verbose = 2
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

all_electrodes_across_experiments = [
    "-(14,)+(6, 22)", "-(3,)+(2,)", "-(27,)+(19,)",
    "-(27,)+(26,)", "-(139,)+(131,)", "-(136,)+(144,)",
    "-(131,)+(130,)", "-(155,)+(154,)"]

if folder_name in ['Day7_AM', 'Day8_AM']:
    # for day8_AM block 4:
    # stim_info_df.index[stim_info_df.index > 30800000]
    # plt.plot(stim_info_df.index, stim_info_df.index ** 0, 'o')
    has_audible_timing = False
    legend_halfway_idx = 2
    which_outcome = 'angle'
    color_background = True
    emg_label_subset = ['LVL', 'LMH', 'LTA', 'LMG', 'RLVL', 'RMH', 'RTA', 'RMG']
    only_these_electrodes = ["-(14,)+(6, 22)", "-(3,)+(2,)"]
    stim_ax_height = 0.5 * len(only_these_electrodes)
    # emg_ax_height = 14 - stim_ax_height
    control_label_subset = [
        ('LeftElbow', 'angle'), ('RightElbow', 'angle'),
    ]
    outcome_label_subset = [
        ('LeftKnee', 'angle'), ('RightKnee', 'angle'),
    ]
    emg_zoom = 1.5
    baseline_timestamps = [int(1e6 + 250), int(1e6 + 250)]
    # 32044466, 99050633
    align_timestamps = [baseline_timestamps[0], 30836967]
    vspan_limits = [
        # left
        [
            (left_sweep * 1e-6, 3.05),
            (3.05, 7.40),
            (7.40, right_sweep * 1e-6)
        ],
        # right
        [
            (left_sweep * 1e-6, 1.39),
            (1.39, 7.57),
            (7.57, right_sweep * 1e-6)
        ]
    ]
    condition_name = 'Participant control'
    show_envelope = True
    show_legend = True
elif folder_name in ['Day12_PM']:
    # for day12_PM block 4:
    # stim_info_df.index[stim_info_df.index > 798000000]
    # stim_info_df.index[stim_info_df.index > 76000000]
    # plt.plot(stim_info_df.index, stim_info_df.index ** 0, 'o')
    has_audible_timing = False
    legend_halfway_idx = 2
    emg_label_subset = ['LVL', 'LMH', 'LTA', 'LMG', 'RLVL', 'RMH', 'RTA', 'RMG']
    only_these_electrodes = ["-(14,)+(6, 22)", "-(3,)+(2,)", "-(139,)+(131,)", "-(136,)+(144,)"]
    stim_ax_height = 0.5 * len(only_these_electrodes)
    # emg_ax_height = 14 - stim_ax_height
    which_outcome = 'angle'
    control_label_subset = [
        ('LeftElbow', 'angle'), ('RightElbow', 'angle'),
    ]
    outcome_label_subset = [
        ('LeftKnee', 'angle'), ('RightKnee', 'angle'),
    ]
    which_outcome = 'angle'
    color_background = True

    emg_zoom = 1.5
    baseline_timestamps = [int(60e6 + 8500), int(790e6 + 8500)]
    align_timestamps = [baseline_timestamps[0], 76021433]
    vspan_limits = [
        # right
        [
            (left_sweep * 1e-6, 4.58),  # stim was already going
            (4.58, 14.75),
            (14.75, right_sweep * 1e-6)
        ]
    ]
    condition_name = 'Participant control'
    show_envelope = True
    show_legend = False
elif folder_name in ['Day11_AM']:
    # for day11_AM block 4:
    # stim_info_df.index[stim_info_df.index > 500010033]
    # stim_info_df.index[stim_info_df.index > xxx]
    # plt.plot(stim_info_df.index, stim_info_df.index ** 0, 'o')

    baseline_timestamps = [int(179e6) + 1500, int(179e6) + 1500]
    align_timestamps = [baseline_timestamps[0], 500046533]

    # for day11_AM block 2:
    # stim_info_df.index[stim_info_df.index > 370000000]
    # stim_info_df.index[stim_info_df.index > xxx]
    # plt.plot(stim_info_df.index, stim_info_df.index ** 0, 'o')
    # align_timestamps = [348082033, 370068533]
    # baseline_timestamps = [int(120e6) + 4500, int(120e6) + 4500]

    only_these_electrodes = [
        "-(14,)+(6, 22)", "-(3,)+(2,)", "-(27,)+(19,)", "-(27,)+(26,)", "-(131,)+(130,)", "-(155,)+(154,)"]
    stim_ax_height = 0.5 * len(only_these_electrodes)
    # emg_ax_height = 14 - stim_ax_height
    has_audible_timing = True

    fps = 29.97
    audible_timing_path = Path(
        f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/6_Video/Day11_AM_Cam1_GH010630_audible_timings.txt")
    legend_halfway_idx = 4
    emg_label_subset = ['LVL', 'LMH', 'LTA', 'LMG', 'RLVL', 'RMH', 'RTA', 'RMG']
    control_label_subset = [
        ('LeftToe', 'z'), ('RightToe', 'z'),
    ]
    outcome_label_subset = [
        ('LeftKnee', 'angle'), ('RightKnee', 'angle'),
    ]
    which_outcome = 'angle'
    color_background = False
    emg_zoom = 2.
    show_envelope = True
    show_legend = False
    condition_name = 'Treadmill walking'

####################################################
control_label_subset = [
    # shoulder, elbow, wrist
    ['LeftUpperArm', 'LeftElbow', 'LeftForeArm'],
    ['RightUpperArm', 'RightElbow', 'RightForeArm'],
    ]
rom = 30
min_ang = 60
l_norm_angle_treshold = 0.01
r_norm_angle_treshold = 0.01
control_thresholds = [min_ang + rom * l_norm_angle_treshold, min_ang + rom * r_norm_angle_treshold, ]
####################################################

def angle_calc(shoulder_point, elbow_point, wrist_point):
    L_up_vect = shoulder_point - elbow_point
    L_down_vect = wrist_point - elbow_point

    l_x_coords_prod = L_up_vect[0] * L_down_vect[0]
    l_y_coords_prod = L_up_vect[1] * L_down_vect[1]

    l_dot_prod = l_x_coords_prod + l_y_coords_prod
    l_up_vec_mag = np.sqrt(L_up_vect[0] ** 2 + L_up_vect[1] ** 2)
    l_down_vec_mag = np.sqrt(L_down_vect[0] ** 2 + L_down_vect[1] ** 2)
    dot_prod = l_dot_prod / (l_up_vec_mag * l_down_vec_mag)
    l_ang = np.degrees(np.arccos(dot_prod))
    return l_ang

def angle_calc_vg(
        shoulder_point, elbow_point, wrist_point,
        xy_only=False):
    vec1 = shoulder_point - elbow_point
    vec2 = wrist_point - elbow_point
    if xy_only:
        vec1[:, 2] = 0.
        vec2[:, 2] = 0.
    return vg.angle(vec1, vec2)

def adjust_channel_name(cn):
    signal_type, ch_num_str = cn.split(' ')
    elec = int(ch_num_str)
    if elec < 128:
        return f"ch {elec} (caudal)"
    else:
        return f"ch {elec - 128} (rostral)"

parquet_folder = data_path / "parquets"
ripple_timecode_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_ripple_timecode_df.parquet"
nf7_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_nf7_df.parquet"
ns5_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_ns5_df.parquet"
emg_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_emg_df.parquet"
stim_info_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_stim_info_df.parquet"
stim_info_traces_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_stim_info_traces_df.parquet"
nev_spikes_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_nev_spikes_df.parquet"
points_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_points_df.parquet"
audible_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_audible_timings_df.parquet"

emg_df = pd.read_parquet(emg_parquet_path)
stim_info_df = pd.read_parquet(stim_info_parquet_path)
nev_spikes = pd.read_parquet(nev_spikes_parquet_path)
points_df = pd.read_parquet(points_parquet_path)
stim_info_trace_df = pd.read_parquet(stim_info_traces_parquet_path)
if has_audible_timing:
    audible_timing = pd.read_parquet(audible_parquet_path)

if standardize_emg:
    emg_df.loc[:, :] = scaler.transform(emg_df)
emg_df.drop(['L Forearm', 'R Forearm', 'Sync'], axis='columns', inplace=True)

determine_side = lambda x: 'L.' if x[0] == 'L' else 'R.'

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

filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), emg_sample_rate)
emg_df = pd.DataFrame(
    signal.sosfiltfilt(filterCoeffs, (emg_df - emg_df.mean()), axis=0),
    index=emg_df.index, columns=emg_df.columns)

vspan_alpha = 0.25

def parse_json_colors(x):
    if x['palette_type'] == 'palettable':
        module_str = 'palettable.' + x['palettable_module']
        palette_module = eval(module_str)
        return palette_module.mpl_colors[x['palette_idx']]
    elif x['palette_type'] == 'hex':
        return mpl.colors.to_rgb(x['palette_idx'])

elec_format_df = pd.read_json('./elec_format_info.json', orient='records')

elec_format_df.loc[:, 'which_array'] = elec_format_df.loc[:, 'which_array'].str.replace(' ', '')
stim_palette = elec_format_df.apply(parse_json_colors, axis='columns').to_list()

notable_electrode_labels = elec_format_df['label'].to_list()
electrode_functional_names = {
    rw['label']: rw['pretty_label'] for rw_idx, rw in elec_format_df.iterrows()
}
electrode_which_array = {
    rw['label']: rw['which_array'] for rw_idx, rw in elec_format_df.iterrows()
}
base_electrode_hue_map = {
    nm: c for nm, c in zip(notable_electrode_labels, stim_palette)
}

vspan_palette = palettable.colorbrewer.qualitative.Pastel2_3.mpl_colors
vspan_colors = [
    # right
    [vspan_palette[0], vspan_palette[1], vspan_palette[0]],
]

dummy_stim_info_entry = {
    'elecCath': (0,),
     'elecAno': (0,),
     'amp': 0,
     'freq': 0,
     'pulseWidth': 0,
     'isContinuous': 0.,
     'res': 3,
     'nipTime': 0,
     'time': baseline_timestamps[0] * 1e-6,
     'original_timestamp_usec': baseline_timestamps[0],
     'delta_timestamp_usec': 0,
     'elecConfig_str': '-(0,)+(0,)',
     'elecAll': (0,)
    }
stim_info_df.loc[baseline_timestamps[0], :] = dummy_stim_info_entry

if show_envelope:
    envelope_df = pd.DataFrame(emg_df ** 2, index=emg_df.index, columns=emg_df.columns)
    window = int((filterOptsEnvelope['low']['Wn'] ** -1) / (emg_sample_rate ** -1))
    envelope_df = envelope_df.rolling(window, center=True).mean().fillna(0)
    envelope_df.loc[:, :] = np.sqrt(envelope_df.to_numpy())
    envelope_df.fillna(0, inplace=True)

all_emg_dict = {}
all_kin_dict = {}
all_spikes_dict = {}
if has_audible_timing:
    all_audible_dict = {}
if show_envelope:
    all_envelope_dict = {}
metadata_fields = ['amp', 'freq', 'elecConfig_str']

for ctrl_idx, ctrl_dim in enumerate(control_label_subset):
    shoulder = points_df.xs(ctrl_dim[0], axis='columns', level='label').loc[:, ('x', 'y', 'z')]
    elbow = points_df.xs(ctrl_dim[1], axis='columns', level='label').loc[:, ('x', 'y', 'z')]
    wrist = points_df.xs(ctrl_dim[2], axis='columns', level='label').loc[:, ('x', 'y', 'z')]
    ang = pd.Series(angle_calc(shoulder.T.to_numpy(), elbow.T.to_numpy(), wrist.T.to_numpy()), index=points_df.index)
    ang_vg = pd.Series(
        angle_calc_vg(
            shoulder.to_numpy(), elbow.to_numpy(), wrist.to_numpy(),
            xy_only=True),
        index=points_df.index)
    ang_ctrl_raw = pd.Series(1 - ((ang - min_ang) / rom), index=points_df.index)
    ang_ctrl = pd.Series(np.clip(ang_ctrl_raw, 0, 1) - 0.5, index=points_df.index)
    thresh = 0.01  #  thresh = control_thresholds[ctrl_idx]
    crossings = ang_ctrl.index[(ang_ctrl >= thresh) & (ang_ctrl.shift(1) < thresh)]
    this_amp_trace = stim_info_trace_df.loc[:, ('amp', '-(3,)+(2,)')]
    amp_thresh = 100
    amp_mask_falling = (this_amp_trace >= amp_thresh) & (this_amp_trace.shift(1) < amp_thresh)
    amp_mask_rising = (this_amp_trace <= amp_thresh) & (this_amp_trace.shift(1) > amp_thresh)
    amp_crossings = this_amp_trace.index[amp_mask_falling | amp_mask_rising]
    amp_crossings_points, _ = closestSeries(referenceIdx=pd.Series(amp_crossings), sampleFrom=pd.Series(ang.index), strictly='neither')
    fig, ax = plt.subplots(figsize=(3, 2))
    ax2 = ax.twinx()
    ax.plot(points_df.index * 1e-6, ang, label='control angle')
    ax.plot(points_df.index * 1e-6, ang_vg, label='control angle (vg)')
    ax.plot(amp_crossings_points * 1e-6, ang.loc[amp_crossings_points], '*', markersize=5)
    ax.legend(loc='upper right')
    ax.set_ylabel('angle (deg.)')
    ax2.plot(this_amp_trace.index * 1e-6, this_amp_trace * 1e-3, c='r', label='stim amp')
    ax2.set_ylabel('amp (mA)')
    ax2.legend(loc='lower right')


points_epochs = {}

train_thresh = .3
for name, group in nev_spikes.groupby('elecConfig_str'):
    print(f'{name}:')
    time_since = group['time_seconds'].diff()
    time_since.iloc[0] = np.inf
    these_first_timestamps = group.loc[time_since > train_thresh, 'time_seconds']
    time_after = group['time_seconds'].diff(periods=-1) * -1
    time_after.iloc[-1] = np.inf
    these_last_timestamps = group.loc[time_after > train_thresh, 'time_seconds']
    points_first_ts, _ = closestSeries(group.loc[these_first_timestamps.index, 'time_usec'], pd.Series(points_df.index))
    points_last_ts, _ = closestSeries(group.loc[these_last_timestamps.index, 'time_usec'], pd.Series(points_df.index))
    if len(these_first_timestamps):
        these_first_timestamps_str = ", ".join([f"{ts:.3f} ({points_first_ts.to_list()[idx]})" for idx, ts in enumerate(these_first_timestamps)])
        these_last_timestamps_str = ", ".join([f"{ts:.3f} ({points_last_ts.to_list()[idx]})" for idx, ts in enumerate(these_last_timestamps)])
        print(f"\tFirst: {these_first_timestamps_str}\n\t Last: {these_last_timestamps_str}")
        points_epochs[name] = list(zip(points_first_ts.to_list(), points_last_ts.to_list()))

aligned_kin_dict = {}
for elecConfig, (first_ts, last_ts) in points_epochs.items():



pretty_points_label_lookup = {
    'LeftToe': 'L. toe', 'RightToe': 'R. toe',
    'LeftKnee': 'L. knee', 'RightKnee': 'R. knee',
    'LeftElbow': 'L. elbow', 'RightElbow': 'R. elbow',
    'LeftAnkle': 'L. ankle', 'RightAnkle': 'R. ankle',
    'RightFoot': 'R. foot', 'LeftFoot': 'L. foot',
    'LeftLowerLeg': 'L. shin', 'RightLowerLeg': 'R. shin',
    'LeftUpperLeg': 'L. thigh', 'RightUpperLeg': 'R. thigh',
}

prettify_points_label = lambda x: f'{pretty_points_label_lookup.get(x["label"])} ({x["axis"]})'
prettify_points_label_tuple = lambda x: f'{pretty_points_label_lookup.get(x[0])} ({x[1]})'
