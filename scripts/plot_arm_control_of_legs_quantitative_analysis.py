import matplotlib as mpl
import os

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

if 'CCV_HEADLESS' in os.environ:
    mpl.use('Agg')  # generate postscript output
else:
    mpl.use('QT5Agg')  # generate interactive output

import palettable
import traceback
from isicpy.utils import makeFilterCoeffsSOS, closestSeries
from isicpy.lookup_tables import emg_montages, kinematics_offsets, muscle_names, emg_palette, emg_hue_map, video_info
from pathlib import Path
import pandas as pd
import numpy as np
import cloudpickle as pickle
import pingouin as pg
from scipy.interpolate import pchip_interpolate
import pdb

import gc
import vg
from sklearn.preprocessing import minmax_scale
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
    'lines.markersize': 4,
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
    "xtick.major.size": 1.5,
    "ytick.major.size": 1.5,
    "xtick.minor.size": 0.5,
    "ytick.minor.size": 0.5,
    "xtick.direction": 'in',
    "ytick.direction": 'in',
    #
    "axes.titlepad": 2.,
    "axes.labelpad": 2.,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
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

plots_dt = int(1e3)  # usec

# folder_name = "Day7_AM"
# block_idx = 4
# this_emg_montage = emg_montages['lower']

# folder_name = "Day8_AM"
# block_idx_list = [2, 3, 4]
# this_emg_montage = emg_montages['lower']
#
# #
#
# folder_name = "Day11_AM"
# block_idx = 4
# this_emg_montage = emg_montages['lower_v2']

folder_name = "Day12_PM"
block_idx_list = [4]
this_emg_montage = emg_montages['lower_v2']

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

all_electrodes_across_experiments = [
    "-(14,)+(6, 22)", "-(3,)+(2,)", "-(27,)+(19,)",
    "-(27,)+(26,)", "-(139,)+(131,)", "-(136,)+(144,)",
    "-(131,)+(130,)", "-(155,)+(154,)"]

if folder_name in ['Day7_AM', 'Day8_AM']:
    # example_indices = [
    #     0, 64, 16, 24,
    #     120, 140, 98, 134]
    example_indices = [
        0, 1, 2, 3, 4, 5, 6, 7]

    display_indices = False
    num_example_cols = 4
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
    # example_indices = [
    #     0, 40, 22, 26,
    #     8, 46, 16, 54]
    example_indices = [
        0, 1, 2, 3, 4, 5, 6, 7]
    display_indices = False
    num_example_cols = 4
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
    example_indices = [0, 1, 2, 3]
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

def parse_json_colors(x):
    if x['palette_type'] == 'palettable':
        module_str = 'palettable.' + x['palettable_module']
        palette_module = eval(module_str)
        return palette_module.mpl_colors[x['palette_idx']]
    elif x['palette_type'] == 'hex':
        return mpl.colors.to_rgb(x['palette_idx'])


kin_format_df = pd.read_json('./kin_format_info.json', orient='records')
kin_format_df.loc[:, 'color'] = kin_format_df.apply(parse_json_colors, axis='columns').to_list()

elec_format_df = pd.read_json('./elec_format_info.json', orient='records')
elec_format_df.loc[:, 'which_array'] = elec_format_df.loc[:, 'which_array'].str.replace(' ', '')
elec_format_df.loc[:, 'color'] = elec_format_df.apply(parse_json_colors, axis='columns').to_list()
notable_electrode_labels = elec_format_df['label'].to_list()
electrode_functional_names = {
    rw['label']: rw['pretty_label'] for rw_idx, rw in elec_format_df.iterrows()
}
electrode_which_array = {
    rw['label']: rw['which_array'] for rw_idx, rw in elec_format_df.iterrows()
}
base_electrode_hue_map = {
    nm: c for nm, c in zip(notable_electrode_labels, elec_format_df['color'])
}


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


config_ctrl_lookup = {
    '-(14,)+(6, 22)': {'control': ('LeftElbow', 'angle'), 'target': ('LeftKnee', 'angle'),
                       'not_target': ('RightKnee', 'angle')},
    '-(3,)+(2,)': {'control': ('LeftElbow', 'angle'), 'target': ('LeftKnee', 'angle'),
                   'not_target': ('RightKnee', 'angle')},
    ##  '-(27,)+(19,)': {'control': ('RightElbow', 'angle'), 'target': ('RightKnee', 'angle'),
    ##                   'not_target': ('LeftKnee', 'angle')},
    ##  '-(27,)+(26,)': {'control': ('RightElbow', 'angle'), 'target': ('RightKnee', 'angle'),
    ##                   'not_target': ('LeftKnee', 'angle')},
}

'''
config_ctrl_lookup = {
    '-(14,)+(6, 22)': {'control': ('LeftElbow', 'angle'), 'target': ('LeftToe', 'displacement'), 'not_target': ('RightToe', 'displacement')},
    '-(3,)+(2,)': {'control': ('LeftElbow', 'angle'), 'target': ('LeftToe', 'displacement'), 'not_target': ('RightToe', 'displacement')},
    # '-(14,)+(6, 22)': {'control': ('LeftElbow', 'angle'), 'target': ('LeftLimb', 'length'), 'not_target': ('RightLimb', 'length')},
    # '-(3,)+(2,)': {'control': ('LeftElbow', 'angle'), 'target': ('LeftLimb', 'length'), 'not_target': ('RightLimb', 'length')},
    # '-(27,)+(19,)': {'control': ('RightElbow', 'angle'), 'target': ('RightKnee', 'angle'), 'not_target': ('LeftKnee', 'angle')},
    # '-(27,)+(26,)': {'control': ('RightElbow', 'angle'), 'target': ('RightKnee', 'angle'), 'not_target': ('LeftKnee', 'angle')},
    }

'''

stats_across_list = []
kin_across_list = []
block_delay = 0.

for block_idx in block_idx_list:

    if standardize_emg:
        emg_scaler_path = data_path / "pickles" / "emg_scaler.p"
        with open(emg_scaler_path, 'rb') as handle:
            scaler = pickle.load(handle)

    file_path = data_path / f"Block{block_idx:0>4d}_Synced_Session_Data.mat"

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

    recalc_angles = False
    if recalc_angles:
        for angle_name, angle_labels in angles_dict.items():
            try:
                point_0 = points_df.xs(angle_labels[0], axis='columns', level='label').loc[:, ('x', 'y', 'z')]
                point_1 = points_df.xs(angle_labels[1], axis='columns', level='label').loc[:, ('x', 'y', 'z')]
                point_2 = points_df.xs(angle_labels[2], axis='columns', level='label').loc[:, ('x', 'y', 'z')]
                points_df.loc[:, (angle_name, 'angle')] = angle_calc_vg(
                    point_0.to_numpy(), point_1.to_numpy(), point_2.to_numpy(),
                    xy_only=False)
            except Exception:
                print(f'\nSkipping angle calculation for {angle_name}\n')
                traceback.print_exc()
    ###
    # try:
    #     this_kin_offset = kinematics_offsets[folder_name][block_idx]
    # except Exception:
    #     this_kin_offset = 0
    # points_df.index = points_df.index + int(this_kin_offset * 1e6)
    ###

    stim_info_trace_df = pd.read_parquet(stim_info_traces_parquet_path)
    if has_audible_timing:
        audible_timing = pd.read_parquet(audible_parquet_path)
    '''
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
    '''

    # all_emg_dict = {}
    # all_kin_dict = {}
    # all_spikes_dict = {}
    # metadata_fields = ['amp', 'freq', 'elecConfig_str']

    # identify kinematics thresh crossings
    for ctrl_idx, ctrl_dim in enumerate(control_label_subset):
        shoulder = points_df.xs(ctrl_dim[0], axis='columns', level='label').loc[:, ('x', 'y', 'z')]
        elbow = points_df.xs(ctrl_dim[1], axis='columns', level='label').loc[:, ('x', 'y', 'z')]
        wrist = points_df.xs(ctrl_dim[2], axis='columns', level='label').loc[:, ('x', 'y', 'z')]
        ang = pd.Series(angle_calc(shoulder.T.to_numpy(), elbow.T.to_numpy(), wrist.T.to_numpy()),
                        index=points_df.index)
        ang_vg = pd.Series(
            angle_calc_vg(
                shoulder.to_numpy(), elbow.to_numpy(), wrist.to_numpy(),
                xy_only=True),
            index=points_df.index)
        ang_ctrl_raw = pd.Series(1 - ((ang - min_ang) / rom), index=points_df.index)
        ang_ctrl = pd.Series(np.clip(ang_ctrl_raw, 0, 1) - 0.5, index=points_df.index)
        thresh = 0.01  # thresh = control_thresholds[ctrl_idx]
        crossings = ang_ctrl.index[(ang_ctrl >= thresh) & (ang_ctrl.shift(1) < thresh)]
        this_amp_trace = stim_info_trace_df.loc[:, ('amp', '-(3,)+(2,)')]
        amp_thresh = 100
        amp_mask_falling = (this_amp_trace >= amp_thresh) & (this_amp_trace.shift(1) < amp_thresh)
        amp_mask_rising = (this_amp_trace <= amp_thresh) & (this_amp_trace.shift(1) > amp_thresh)
        amp_crossings = this_amp_trace.index[amp_mask_falling | amp_mask_rising]
        amp_crossings_points, _ = closestSeries(referenceIdx=pd.Series(amp_crossings), sampleFrom=pd.Series(ang.index),
                                                strictly='neither')
        plot_amp_crossings = False
        if plot_amp_crossings:
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
        points_first_ts, _ = closestSeries(group.loc[these_first_timestamps.index, 'time_usec'],
                                           pd.Series(points_df.index))
        points_last_ts, _ = closestSeries(group.loc[these_last_timestamps.index, 'time_usec'],
                                          pd.Series(points_df.index))
        if len(these_first_timestamps):
            these_first_timestamps_str = ", ".join(
                [f"{ts:.3f} ({points_first_ts.to_list()[idx]})" for idx, ts in enumerate(these_first_timestamps)])
            these_last_timestamps_str = ", ".join(
                [f"{ts:.3f} ({points_last_ts.to_list()[idx]})" for idx, ts in enumerate(these_last_timestamps)])
            print(f"\tFirst: {these_first_timestamps_str}\n\t Last: {these_last_timestamps_str}")
            points_epochs[name] = list(zip(points_first_ts.to_list(), points_last_ts.to_list()))

    # fix angle signs
    change_angle_sign = ((folder_name == 'Day8_AM') & (block_idx in [1, 2]))
    if change_angle_sign:
        points_df.loc[:, ('LeftKnee', 'angle')] = 180 - points_df[('LeftKnee', 'angle')]
    change_angle_sign = ((folder_name == 'Day8_AM') & (block_idx in [1, 2, 3, 4]))
    if change_angle_sign:
        points_df.loc[:, ('RightKnee', 'angle')] = 180 - points_df[('RightKnee', 'angle')]

    all_kin_list = []
    all_stats_list = []
    min_duration = 1.  # seconds
    for elecConfig, timestamps in points_epochs.items():
        if elecConfig not in config_ctrl_lookup:
            continue
        these_stats_list = []
        these_interp_data_list = []
        for (first_ts, last_ts) in timestamps:
            if (last_ts - first_ts) < min_duration * 1e6:
                continue
            this_mask = (points_df.index >= first_ts) & (points_df.index < last_ts)
            this_data = pd.concat({
                'control': points_df.loc[this_mask, config_ctrl_lookup[elecConfig]['control']].copy(),
                'target': points_df.loc[this_mask, config_ctrl_lookup[elecConfig]['target']].copy(),
                'not_target': points_df.loc[this_mask, config_ctrl_lookup[elecConfig]['not_target']].copy()
            }, axis='columns')
            stats_df = pg.corr(this_data['control'], this_data['target'])
            stats_df.loc[:, 'first_timestamp'] = first_ts
            stats_df.loc[:, 'last_timestamp'] = last_ts
            stats_df.loc[:, 'elecConfig_str'] = elecConfig
            stats_df.loc[:, 'target_type'] = 'target'
            these_stats_list.append(stats_df)
            alt_stats_df = pg.corr(this_data['control'], this_data['not_target'])
            alt_stats_df.loc[:, 'first_timestamp'] = first_ts
            alt_stats_df.loc[:, 'last_timestamp'] = last_ts
            alt_stats_df.loc[:, 'elecConfig_str'] = elecConfig
            alt_stats_df.loc[:, 'target_type'] = 'not_target'
            these_stats_list.append(alt_stats_df)
            ####
            this_data.index = this_data.index - this_data.index[0]
            new_indices = np.linspace(0, this_data.index[-1], 100, dtype=int)
            interp_data = pchip_interpolate(this_data.index, this_data.to_numpy(), new_indices)
            interp_data_df = pd.DataFrame(interp_data, columns=['control', 'target', 'not_target'])
            interp_data_df.loc[:, 'first_timestamp'] = first_ts
            interp_data_df.loc[:, 'last_timestamp'] = last_ts
            interp_data_df.loc[:, 'elecConfig_str'] = elecConfig
            these_interp_data_list.append(interp_data_df)
            interp_data_df.loc[:, ['control', 'target']] = minmax_scale(interp_data_df.loc[:, ['control', 'target']])
            plot_this_round = False
            if plot_this_round:
                fig, ax = plt.subplots()
                ax.plot(this_data['control'], label=config_ctrl_lookup[elecConfig]['control'])
                ax.plot(new_indices, interp_data[:, 0], label=f"{config_ctrl_lookup[elecConfig]['control']} (interp)")
                ax.plot(this_data['target'], label=config_ctrl_lookup[elecConfig]['target'])
                ax.plot(new_indices, interp_data[:, 1], label=f"{config_ctrl_lookup[elecConfig]['target']} (interp)")
                ax.legend()
        these_stats = pd.concat(these_stats_list).reset_index()
        these_stats.rename(columns={'index': 'type'}, inplace=True)
        all_stats_list.append(these_stats)
        these_interp_data = pd.concat(these_interp_data_list).reset_index()
        these_interp_data.rename(columns={'index': 'normalized_time'}, inplace=True)
        all_kin_list.append(these_interp_data)

    stats_this_block = pd.concat(all_stats_list)
    stats_this_block.loc[:, 'block'] = block_idx
    stats_this_block.loc[:, 'experiment_session'] = f"{folder_name}_block_{block_idx}"
    stats_this_block.loc[:, 'first_timestamp'] += block_delay
    stats_this_block.loc[:, 'last_timestamp'] += block_delay
    stats_across_list.append(stats_this_block)

    kin_this_block = pd.concat(all_kin_list)
    kin_this_block.loc[:, 'block'] = block_idx
    kin_this_block.loc[:, 'experiment_session'] = f"{folder_name}_block_{block_idx}"
    kin_this_block.loc[:, 'first_timestamp'] += block_delay
    kin_this_block.loc[:, 'last_timestamp'] += block_delay
    kin_across_list.append(kin_this_block)

    block_delay += points_df.index[-1]

stats_df = pd.concat(stats_across_list)
kin_df = pd.concat(kin_across_list)

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

elec_format_df.loc[:, 'pretty_label_w_array'] = elec_format_df.apply(
    lambda x: f"{x['pretty_label']}\n({x['which_array']})", axis='columns')
stats_df.loc[:, 'elecConfig_pretty'] = stats_df['elecConfig_str'].map(
    elec_format_df.set_index('label')['pretty_label_w_array'])
stats_df.loc[:, 'time'] = stats_df['first_timestamp'] * 1e-6
stats_df = stats_df.reset_index()
stats_df.loc[:, 'isin_examples'] = stats_df.index.isin(example_indices)
example_timestamps = stats_df.loc[example_indices, 'first_timestamp']


def add_swarm(
        data=None,
        facetgrid=None, palette=None,
        swarmplot_kwargs=dict(),
        label=None, color=None, ):
    ax = plt.gca()
    x_var = facetgrid._x_var
    y_var = facetgrid._y_var
    sns.swarmplot(
        data=data, x=x_var, y=y_var,
        hue=x_var, palette=palette,
        ax=ax, **swarmplot_kwargs)
    ax.set_ylim([-1, 1])
    # print('\n'.join(dir(facetgrid)))
    return


swarmplot_kwargs = dict(
    edgecolor='k',
    linewidth=0.25,
    size=2.5, facecolor='k'
)

this_palette = (
    elec_format_df
    .loc[elec_format_df['label'].isin(stats_df['elecConfig_str']), :]
    .set_index('pretty_label_w_array')['color']
    .to_dict()
)

block_list_string = '_'.join([f"{bidx}" for bidx in block_idx_list])
pdf_path = pdf_folder / Path(f"{folder_name}_Blocks_{block_list_string}_arm_ctrl_legs_quantitative.pdf")
show_plots = True
with PdfPages(pdf_path) as pdf:
    w, h = 2.6, 1.2

    # stats_df.loc[stats_df['target_type'] == 'target', :].groupby('elecConfig_pretty').median()['r']
    g = sns.catplot(
        data=stats_df.loc[stats_df['target_type'] == 'target', :],
        x='elecConfig_pretty', y='r',
        # col='experiment_session',
        palette=this_palette,
        kind='box', height=h, aspect=w / h,
        width=0.6, fliersize=.1
    )
    g.map_dataframe(add_swarm, facetgrid=g, palette=this_palette, swarmplot_kwargs=swarmplot_kwargs)
    g.set_ylabels('Correlation\n(a.u.)')
    g.set_xlabels('Stim. electrode configuration')
    g.figure.suptitle('Elbow-knee angle correlation')
    g.figure.align_labels()
    g.figure.set_size_inches(1.6, 1.)
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()

    '''
    g = sns.relplot(
        data=kin_df, x='normalized_time', y='target',
        row='elecConfig_str', col='experiment_session',
        kind='line',
        estimator=None, units='first_timestamp', hue='first_timestamp',
        height=3, aspect=1
        )
    g.figure.align_labels()
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()

        g = sns.relplot(
            data=kin_df, x='normalized_time', y='control',
            row='elecConfig_str', col='experiment_session',
            kind='line',
            estimator=None, units='first_timestamp', hue='first_timestamp',
            height=3, aspect=1
            )
        g.figure.align_labels()
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if show_plots:
            plt.show()
        else:
            plt.close()

        # stats_df.loc[stats_df['target_type'] == 'target', :].sort_values('r').iloc[0, :]
        # plt.scatter(stats_df.loc[stats_df['target_type'] == 'target', 'first_timestamp'] * 1e-6, stats_df.loc[stats_df['target_type'] == 'target', 'r'])
        '''

    w, h = 3.6, 2

    sns.set(
        context='paper', style='white',
        palette='dark', font='sans-serif',
        font_scale=1, color_codes=True, rc=snsRCParams
    )

    fig = plt.figure(figsize=(w, h))
    gs = fig.add_gridspec(
        3, num_example_cols + 1,
        height_ratios=(1.5, 1., 1.), width_ratios=[1] * num_example_cols + [1.],
        left=0.025, right=0.975, bottom=0.025, top=0.925,
        wspace=0.1, hspace=0.3
    )
    corr_ax = fig.add_subplot(gs[0, :-1])
    sns.scatterplot(
        data=stats_df.loc[stats_df['target_type'] == 'target', :],
        x='time', y='r', hue='elecConfig_pretty',
        palette=this_palette, ax=corr_ax)
    corr_ax.plot(
        stats_df.loc[stats_df.index.isin(example_indices), 'time'],
        stats_df.loc[stats_df.index.isin(example_indices), 'r'],
        'o', markeredgewidth=0.5, fillstyle='none', markeredgecolor='k',
    )
    if display_indices:
        for row_idx, row in stats_df.loc[stats_df['target_type'] == 'target', :].iterrows():
            corr_ax.text(row['time'], row['r'], f"{row_idx}")

    corr_ax.set_ylabel('Correlation (a.u.)')
    corr_ax.set_xlabel('Time (sec.)')
    custom_lines = []
    custom_text = []
    for key, value in this_palette.items():
        custom_lines.append(
            Line2D(
                [0], [0],
                marker='o', markerfacecolor=value, lw=0,
                markeredgewidth=0))
        custom_text.append(key)
    corr_ax.legend(
        custom_lines, custom_text, loc='center left',
        bbox_to_anchor=(1.025, .5), borderaxespad=0., title='Stim.\nelectrode\nconfiguration')

    ax_list = (
            [fig.add_subplot(gs[1, idx]) for idx in range(num_example_cols)] +
            [fig.add_subplot(gs[2, idx]) for idx in range(num_example_cols)]
    )
    # ax2_list = [ax.twinx() for ax in ax_list]

    all_ax_ylim = []
    # all_ax2_ylim = []

    which_linestyle = {
        'RightElbow': '--',
        'RightKnee': '--',
        'LeftElbow': '-',
        'LeftKnee': '-',
    }
    for idx, ax in enumerate(ax_list):
        this_kin = kin_df.loc[kin_df['first_timestamp'] == example_timestamps.iloc[idx], :]
        t = np.linspace(this_kin['first_timestamp'].iloc[0] * 1e-6, this_kin['last_timestamp'].iloc[0] * 1e-6,
                        this_kin.shape[0])
        t = t - t[0]
        this_corr = stats_df.iloc[example_indices[idx], :]['r']

        control_label, control_axis = config_ctrl_lookup[stats_df.iloc[example_indices[idx], :]['elecConfig_str']][
            'control']
        target_label, target_axis = config_ctrl_lookup[stats_df.iloc[example_indices[idx], :]['elecConfig_str']][
            'target']

        target_pretty_label = f"{pretty_points_label_lookup[target_label]} {target_axis} (deg.)"
        control_pretty_label = f"{pretty_points_label_lookup[control_label]} {control_axis} (deg.)"
        # control_mask = (kin_format_df['label'] == control_label) & (kin_format_df['axis'] == control_axis)
        # target_mask = (kin_format_df['label'] == target_label) & (kin_format_df['axis'] == target_axis)
        # control_color = kin_format_df.loc[control_mask, 'color'].iloc[0]
        # target_color = kin_format_df.loc[target_mask, 'color'].iloc[0]

        control_color = '#EE7733'
        target_color = '#BBBBBB'

        control_ls = which_linestyle[control_label]
        target_ls = which_linestyle[target_label]

        ax.plot(
            t, this_kin['control'], c=control_color, ls=control_ls,
            label=pretty_points_label_lookup[control_label])

        all_ax_ylim.append(ax.get_ylim())
        if idx not in [0, 4]:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Normalized\nangle (a.u.)')
        if idx < 4:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Time (sec.)')
        # ax2 = ax2_list[idx]

        ax.plot(t, this_kin['target'], c=target_color, ls=target_ls,
                label=pretty_points_label_lookup[target_label])

        # all_ax2_ylim.append(ax2.get_ylim())
        # if idx not in [3, 7]:
        #     ax2.set_yticklabels([])
        # else:
        #     ax2.set_ylabel('Normalized knee angle (a.u.)')
        if idx == 3:
            custom_lines = [
                Line2D([0], [0], color=control_color, ls='-', lw=1),
                Line2D([0], [0], color=target_color, ls='-', lw=1),
                Line2D([0], [0], color=control_color, ls='--', lw=1),
                Line2D([0], [0], color=target_color, ls='--', lw=1),
            ]
            custom_text = [
                'Left elbow',
                'Left knee',
                'Right elbow',
                'Right knee',
            ]
            ax.legend(
                custom_lines, custom_text, title='Joint',
                loc='center left', bbox_to_anchor=(1.05, 0.), borderaxespad=0., )

    for idx, ax in enumerate(ax_list):
        new_ymin, new_ymax = min([l[0] for l in all_ax_ylim]), max([l[1] for l in all_ax_ylim])
        new_mean_y = (new_ymin + new_ymax) / 2
        new_y_extent = (new_ymax - new_ymin) / 2

        # ax.tick_params(axis='y', labelrotation=-45, pad=0)
        ax.set_ylim([new_mean_y - 1.1 * new_y_extent, new_mean_y + 1.1 * new_y_extent])

        # new_ymin2, new_ymax2 = min([l[0] for l in all_ax2_ylim]), max([l[1] for l in all_ax2_ylim])
        # new_mean_y2 = (new_ymin2 + new_ymax2) / 2
        # new_y_extent2 = (new_ymax2 - new_ymin2) / 2
        # ax2 = ax2_list[idx]
        # # ax2.tick_params(axis='y', labelrotation=-45, pad=0)
        # ax2.set_ylim([new_mean_y2 - 1.1 * new_y_extent2, new_mean_y2 + 1.1 * new_y_extent2])

        this_corr = stats_df.iloc[example_indices[idx], :]['r']
        ax.text(.99, .9, f"r = {this_corr:.2f}", transform=ax.transAxes, va='center', ha='right')
        marker_x, marker_y = ax.transLimits.inverted().transform((.1, .9))
        ax.plot(
            marker_x, marker_y, 'o',
            markeredgewidth=0.5,
            markerfacecolor=this_palette[stats_df.iloc[example_indices[idx], :]['elecConfig_pretty']])

    fig.align_labels()
    fig.tight_layout()
    sns.despine(fig=fig)
    fig.suptitle('Elbow-knee angle correlation')
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()
