
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

plots_dt = int(1e3)  # usec

folder_name = "Day8_AM"
block_idx = 4
this_emg_montage = emg_montages['lower']

# folder_name = "Day12_PM"
# block_idx = 4
# this_emg_montage = emg_montages['lower_v2']

# folder_name = "Day11_AM"
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

data_path = Path(f"/users/rdarie/scratch/3_Preprocessed_Data/{folder_name}")

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

if folder_name in ['Day8_AM']:
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

parquet_folder = data_path / "parquets"
emg_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_emg_df.parquet"
stim_info_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_stim_info_df.parquet"
nev_spikes_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_nev_spikes_df.parquet"
points_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_points_df.parquet"
audible_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_audible_timings_df.parquet"
reprocess_raw = False
save_parquets = True

if (not os.path.exists(parquet_folder)) or reprocess_raw:
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
        except Exception:
            print(f'\nSkipping angle calculation for {angle_name}\n')
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
        except Exception:
            print(f'\nSkipping limb length calculation for {length_name}\n')
            traceback.print_exc()

    '''
    points_df.columns = points_df.columns.to_frame().apply(
        lambda x: f"{x.iloc[0]}_{x.iloc[1]}", axis=1).to_list()
    points_df.columns.name = 'label'
    '''

    stim_info_df = data_dict['stim_info']
    stim_info_df.loc[:, 'elecConfig_str'] = stim_info_df.apply(
        lambda x: f'-{x["elecCath"]}+{x["elecAno"]}', axis='columns')

    stim_info_df.loc[:, 'elecAll'] = stim_info_df.apply(lambda x: x['elecCath'] + x['elecAno'], axis='columns')
    nev_spikes = data_dict['ripple']['NEV'].copy()
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

    if has_audible_timing:
        def int_or_nan(x):
            try:
                return int(x)
            except:
                return np.nan

        audible_timing = pd.read_csv(audible_timing_path)
        audible_timing = audible_timing.stack().reset_index().iloc[:, 1:]
        audible_timing.columns = ['words', 'time']
        audible_timing.loc[:, ['m', 's', 'f']] = audible_timing['time'].apply(lambda x: x.split(':')).apply(
            pd.Series).to_numpy()
        audible_timing.loc[:, ['m', 's', 'f']] = audible_timing.loc[:, ['m', 's', 'f']].applymap(int_or_nan)
        audible_timing.loc[:, 'total_frames'] = (
                audible_timing['m'] * 60 * 30 +
                audible_timing['s'] * 30 +
                audible_timing['f'])
        # video_t = audible_timing.loc[:, 'total_frames'].apply(lambda x: pd.Timedelta(x / 29.97, unit='sec'))
        # audible_timing.loc[:, 'timestamp'] = pd.Timestamp(year=2022, month=10, day=31) + video_t
        first_ripple = data_dict['ripple']['TimeCode'].iloc[0, :]
        ripple_origin_timestamp = timestring_to_timestamp(
            first_ripple['TimeString'], day=31, fps=fps, timecode_type='NDF')
        audible_timing.loc[:, 'timedeltas'] = audible_timing.loc[:, 'total_frames'].apply(
            lambda x: pd.Timedelta(x / 29.97, unit='sec'))
        audible_timecodes = pd.Timestamp(year=2022, month=10, day=31) + audible_timing.loc[:, 'timedeltas']
        audible_timing.loc[:, 'ripple_time'] = (audible_timecodes - ripple_origin_timestamp).apply(
            lambda x: x.total_seconds()) + first_ripple['PacketTime']
        audible_timing.dropna(inplace=True)
        audible_timing.loc[:, 'time_usec'] = audible_timing['ripple_time'].apply(lambda x: int(x * 1e6))

    if save_parquets:
        if not os.path.exists(parquet_folder):
            os.makedirs(parquet_folder)
        emg_df.to_parquet(emg_parquet_path)
        stim_info_df.to_parquet(stim_info_parquet_path)
        nev_spikes.to_parquet(nev_spikes_parquet_path)
        points_df.to_parquet(points_parquet_path)
        if has_audible_timing:
            audible_timing.to_parquet(audible_parquet_path)
else:
    emg_df = pd.read_parquet(emg_parquet_path)
    stim_info_df = pd.read_parquet(stim_info_parquet_path)
    nev_spikes = pd.read_parquet(nev_spikes_parquet_path)
    points_df = pd.read_parquet(points_parquet_path)
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

# elec_format_df.loc[:, 'pretty_label'] = elec_format_df.loc[:, 'pretty_label'].str.replace(' ', '')
# stim_palette = [
#     palettable.colorbrewer.qualitative.Paired_12.mpl_colors[idx]
#     for idx in elec_format_df['palette_idx']
# ]

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
    # filterCoeffsEnv = makeFilterCoeffsSOS(filterOptsEnvelope.copy(), emg_sample_rate)
    '''envelope_df = pd.DataFrame(
        signal.sosfiltfilt(filterCoeffsEnv, emg_df.abs(), axis=0),
        index=emg_df.index, columns=emg_df.columns)'''
    '''envelope_df = pd.DataFrame(
        signal.sosfiltfilt(filterCoeffsEnv, emg_df ** 2, axis=0),
        index=emg_df.index, columns=emg_df.columns)
    envelope_df.clip(lower=0, inplace=True)
    envelope_df.loc[:, :] = np.sqrt(envelope_df)'''
    '''envelope_df = pd.DataFrame(
        # np.abs(signal.hilbert(emg_df, axis=0)),
        signal.sosfiltfilt(filterCoeffsEnv, np.abs(signal.hilbert(emg_df, axis=0)), axis=0),
        index=emg_df.index, columns=emg_df.columns)'''
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
    if show_envelope:
        all_envelope_dict[this_entry] = pd.DataFrame(
            envelope_df.loc[this_mask, :].to_numpy(),
            index=emg_epoch_t, columns=envelope_df.columns)
        all_envelope_dict[this_entry].index.name = 'time_usec'

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
    # all_kin_dict[this_entry] = all_kin_dict[this_entry] - points_df.loc[baseline_timestamps[ts_idx], :]
    all_emg_dict[this_entry] = all_emg_dict[this_entry] - emg_df.loc[baseline_timestamps[ts_idx], :]
    if show_envelope:
        all_envelope_dict[this_entry] = all_envelope_dict[this_entry] - emg_df.loc[baseline_timestamps[ts_idx], :]
    # baseline based on first entry
    # all_kin_dict[this_entry] = all_kin_dict[this_entry] - all_kin_dict[this_entry].iloc[0, :]
    # all_emg_dict[this_entry] = all_emg_dict[this_entry] - emg_df.loc[baseline_timestamps[ts_idx], :]
    # if show_envelope:
    #     all_envelope_dict[this_entry] = all_envelope_dict[this_entry] - emg_df.loc[baseline_timestamps[ts_idx], :]

    this_mask = (nev_spikes['time_usec'] >= (timestamp + int(left_sweep))) & (nev_spikes['time_usec'] <= (timestamp + int(right_sweep)))
    all_spikes_dict[timestamp] = nev_spikes.loc[this_mask, :].copy()
    all_spikes_dict[timestamp].loc[:, 'time_usec'] = all_spikes_dict[timestamp]['time_usec'] - timestamp
    if has_audible_timing:
        this_mask = (audible_timing['time_usec'] >= (timestamp + int(left_sweep))) & (audible_timing['time_usec'] <= (timestamp + int(right_sweep)))
        all_audible_dict[timestamp] = audible_timing.loc[this_mask, :].copy()
        all_audible_dict[timestamp].loc[:, 'time_usec'] = all_audible_dict[timestamp]['time_usec'] - timestamp

aligned_emg_df = pd.concat(all_emg_dict, names=['timestamp_usec'] + metadata_fields)
if show_envelope:
    aligned_envelope_df = pd.concat(all_envelope_dict, names=['timestamp_usec'] + metadata_fields)

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
    except Exception:
        print(f'\nSkipping displacement calc for {name}\n')
        traceback.print_exc()

disps_df = pd.concat(displacements_list, axis='columns')
disps_df.columns = pd.MultiIndex.from_tuples(columns_list, names=['label', 'axis'])
aligned_kin_df = pd.concat([aligned_kin_df, disps_df], axis='columns')

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

# kin_format_df = pd.read_csv('./kin_format_info.csv')
# kin_format_df.to_json('./kin_format_info.json', orient='records', indent=4)
kin_format_df = pd.read_json('./kin_format_info.json', orient='records')

kin_format_df.loc[:, 'color'] = kin_format_df.apply(parse_json_colors, axis='columns').to_list()

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
        legend_line.set_linewidth(1)
        handlebox.add_artist(legend_line)
        return legend_line

side_label_offset = 1.01
side_label_opts = dict(
    # fontdict=dict(
    #     size=snsRCParams['legend.fontsize']
    # ),
    ha='left', va='bottom', rotation=315,
    rotation_mode='anchor'
    )

show_raw_emg = True
baseline_dur = 3.
show_plots = True
figsize = (1.6, 4)
with PdfPages(pdf_path) as pdf:
    emg_indexes_to_plot = aligned_emg_df.index[::emg_downsample]
    plot_emg = aligned_emg_df.loc[emg_indexes_to_plot, emg_label_subset].stack().to_frame(name='signal').reset_index()
    plot_emg.loc[:, 'time_sec'] = plot_emg['time_usec'] * 1e-6
    if show_envelope:
        plot_envelope = aligned_envelope_df.loc[emg_indexes_to_plot, emg_label_subset].stack().to_frame(name='signal').reset_index()
        plot_envelope.loc[:, 'time_sec'] = plot_envelope['time_usec'] * 1e-6
    # trim baseline
    trim_mask_emg = (plot_emg['timestamp_usec'] == baseline_timestamps[0]) & (plot_emg['time_sec'] > baseline_dur)
    plot_emg.drop(index=plot_emg.index[trim_mask_emg], inplace=True)
    if show_envelope:
        plot_envelope.drop(index=plot_envelope.index[trim_mask_emg], inplace=True)

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

    # trim baseline
    trim_mask_kin = (plot_kin['timestamp_usec'] == baseline_timestamps[0]) & (plot_kin['time_sec'] > baseline_dur)
    plot_kin.drop(index=plot_kin.index[trim_mask_kin], inplace=True)

    pretty_points_labels = [pretty_points_label_lookup[lbl] for lbl in plot_kin['label'].unique()]
    pretty_outcome_labels = [prettify_points_label_tuple(lbl) for lbl in outcome_label_subset]
    pretty_control_labels = [prettify_points_label_tuple(lbl) for lbl in control_label_subset]

    fig = plt.figure(figsize=figsize)
    if has_audible_timing:
        gs = fig.add_gridspec(
            5, 2, height_ratios=(3, 6, 12, 3, 1.5), width_ratios=(3, 10),  # , 1.2
            left=0.025, right=0.975, bottom=0.025, top=0.975,
            wspace=0.05, hspace=0.025
        )
    else:
        gs = fig.add_gridspec(
            4, 2, height_ratios=(3, 6, 12, 4.5), width_ratios=(3, 10),  # , 1.2
            left=0.025, right=0.975, bottom=0.025, top=0.975,
            wspace=0.05, hspace=0.025
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
    top_ax = kin_ax

    this_emg_palette = [emg_hue_map[nm] for nm in emg_label_subset]
    this_emg_hue_map = {
        nm: emg_hue_map[nm] for nm in emg_label_subset
        }

    if show_raw_emg:
        vert_span = plot_emg['signal'].max() - plot_emg['signal'].min()
        vert_offset = -40e-2 * vert_span
    else:
        vert_span = plot_envelope['signal'].max() - plot_envelope['signal'].min()
        vert_offset = -60e-2 * vert_span
    horz_span = plot_emg['time_sec'].max() - plot_emg['time_sec'].min()
    horz_offset = 0 * horz_span
    n_offsets = 0

    emg_major_yticks = []
    emg_minor_yticks = []
    emg_yticklabels = []
    vert_offset_lookup = {}

    pretty_emg_labels = [f'{determine_side(n)} {muscle_names[n]}' for n in emg_label_subset]
    emg_label_text_ypos = []
    '''
    pretty_emg_labels = [f'         {muscle_names[n]}' for n in emg_label_subset]
    idx_of_half = int(len(emg_label_subset) / 2)
    pretty_emg_labels[0] = f'{determine_side(emg_label_subset[0])} {muscle_names[emg_label_subset[0]]}'
    pretty_emg_labels[idx_of_half] = f'{determine_side(emg_label_subset[idx_of_half])} {muscle_names[emg_label_subset[idx_of_half]]}'
    '''
    for name in emg_label_subset:
        group_index = plot_emg.loc[plot_emg['label'] == name, :].index

        # plot_emg.loc[group_index, 'signal'] = plot_emg.loc[group_index, 'signal'] - plot_emg.loc[group_index, 'signal'].mean()
        # plot_envelope.loc[group_index, 'signal'] = plot_envelope.loc[group_index, 'signal'] - plot_emg.loc[group_index, 'signal'].mean()

        plot_emg.loc[group_index, 'signal'] = emg_zoom * plot_emg.loc[group_index, 'signal'] + n_offsets * vert_offset
        if show_envelope:
            plot_envelope.loc[group_index, 'signal'] = emg_zoom * plot_envelope.loc[group_index, 'signal'] + n_offsets * vert_offset

        emg_label_text_ypos.append(n_offsets * vert_offset)
        plot_emg.loc[group_index, 'time_sec'] = plot_emg.loc[group_index, 'time_sec'] + n_offsets * horz_offset
        plot_envelope.loc[group_index, 'time_sec'] = plot_envelope.loc[group_index, 'time_sec'] + n_offsets * horz_offset

        vert_offset_lookup[name] = n_offsets * vert_offset
        emg_major_yticks += [n_offsets * vert_offset]
        emg_minor_yticks += [(n_offsets - 0.25) * vert_offset, (n_offsets + 0.25) * vert_offset]
        emg_yticklabels += [f"{-0.25 * vert_offset:.2f}", "0.", f"{0.25 * vert_offset:.2f}"]
        n_offsets += 1

    left_mask_emg = plot_emg['timestamp_usec'] == align_timestamps[0]
    right_mask_emg = plot_emg['timestamp_usec'] == align_timestamps[1]

    if show_envelope:
        emg_alpha = 0.4
    else:
        emg_alpha = 1.
    if show_raw_emg:
        sns.lineplot(
            data=plot_emg.loc[left_mask_emg, :], ax=emg_ax[0],
            x='time_sec', y='signal', alpha=emg_alpha,
            hue='label', palette=this_emg_hue_map, legend=False)
        sns.lineplot(
            data=plot_emg.loc[right_mask_emg, :], ax=emg_ax[1],
            x='time_sec', y='signal', alpha=emg_alpha,
            hue='label', palette=this_emg_hue_map, legend=False)
    if show_envelope:
        sns.lineplot(
            data=plot_envelope.loc[left_mask_emg, :], ax=emg_ax[0],
            x='time_sec', y='signal', linewidth=1.,
            hue='label', palette=this_emg_hue_map, legend=False)
        sns.lineplot(
            data=plot_envelope.loc[right_mask_emg, :], ax=emg_ax[1],
            x='time_sec', y='signal', linewidth=1.,
            hue='label', palette=this_emg_hue_map, legend=False)

    time_ticks_left = [0, 2]
    time_ticks_right = [0, 5, 10]
    for this_ax in emg_ax:
        this_ax.set_xlim((left_sweep * 1e-6, right_sweep * 1e-6))
        this_ax.set_xlabel('')
        if this_ax not in bottom_ax:
            plt.setp(this_ax.get_xticklabels(), visible=False)
        this_ax.yaxis.set_major_locator(ticker.MultipleLocator(np.abs(vert_offset)))
        this_ax.yaxis.set_minor_locator(ticker.MultipleLocator(np.abs(vert_offset) / 2))
        plt.setp(this_ax.get_yticklabels(), visible=False)
        sns.despine(ax=this_ax)

    emg_ax[0].xaxis.set_major_locator(ticker.FixedLocator(time_ticks_left))
    emg_ax[0].xaxis.set_minor_locator(ticker.FixedLocator(np.arange(left_sweep * 1e-6, right_sweep * 1e-6, 1)))
    emg_ax[1].xaxis.set_major_locator(ticker.FixedLocator(time_ticks_right))
    emg_ax[1].xaxis.set_minor_locator(ticker.FixedLocator(np.arange(left_sweep * 1e-6, right_sweep * 1e-6, 1)))

    emg_ax[0].set_ylabel('Normalized\nEMG (a.u.)')
    emg_ax[1].set_ylabel('')

    emg_ax_trans = transforms.blended_transform_factory(emg_ax[1].transAxes, emg_ax[1].transData)
    for ii, name in enumerate(emg_label_subset):
        # emg_ax[0].axhline(y=emg_label_text_ypos[ii], color=this_emg_palette[ii])
        # emg_ax[1].axhline(y=emg_label_text_ypos[ii], color=this_emg_palette[ii])
        emg_ax[1].text(
            side_label_offset, emg_label_text_ypos[ii], pretty_emg_labels[ii],
            transform=emg_ax_trans, color=this_emg_palette[ii], **side_label_opts)
    if show_legend:
        dummy_legend_handle = mpl.patches.Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor='none', visible=False)
        custom_legend_lines = [dummy_legend_handle] + [
            Line2D([0], [0], color=c, lw=1)
            for idx, c in enumerate(this_emg_palette)
            ]
        emg_ax[1].legend(custom_legend_lines, ['EMG'] + pretty_emg_labels, loc='center left', bbox_to_anchor=(1.025, .5), borderaxespad=0.)

    # reorder to match left-right order
    ### only those in this experiment, plot_stim_spikes
    # electrode_labels = elec_format_df.loc[elec_format_df['label'].isin(plot_stim_spikes['elecConfig_str']), :].sort_values(['which_array', 'palette_idx'])['label']
    ## all electrodes, across days
    electrode_labels = elec_format_df.loc[elec_format_df['label'].isin(all_electrodes_across_experiments), :].sort_values(['which_array', 'palette_idx'])['label']
    electrode_hue_map = {
        nm: base_electrode_hue_map[nm] for nm in electrode_labels
        }
    pretty_electrode_labels = [f'{electrode_functional_names[cfg]} ({electrode_which_array[cfg]})' for cfg in electrode_labels]

    if has_audible_timing:
        # audible_palette = palettable.colorbrewer.qualitative.Set3_12.mpl_colors
        audible_hue_map = {
            'Left Foot': mpl.colors.to_rgb("#EE7733"),  # audible_palette[5],
            'Right Foot': mpl.colors.to_rgb("#BBBBBB"), # audible_palette[6]
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
        lc.set_linewidth(1.)
        line = verbal_ax[0].add_collection(lc)
        right_mask_audible = (plot_audible['timestamp_usec'] == align_timestamps[1])
        x = plot_audible.loc[right_mask_audible, 'time_usec'].drop_duplicates() * 1e-6
        origins = np.concatenate([x.to_numpy().reshape(1, -1), np.zeros((1, x.shape[0])) - 0.1], axis=0).T
        endpoints = origins.copy()
        endpoints[:, 1] -= .8
        segments = np.concatenate([origins[:, np.newaxis, :], endpoints[:, np.newaxis, :]], axis=1)
        lc = mpl.collections.LineCollection(segments, colors=plot_audible.loc[right_mask_audible, 'words'].map(audible_hue_map).to_list())
        lc.set_linewidth(1.)
        line = verbal_ax[1].add_collection(lc)

        for this_ax in verbal_ax:
            this_ax.set_xlim([left_sweep * 1e-6, right_sweep * 1e-6])
            this_ax.set_ylim([-1., 0.])
            this_ax.set_xlabel('')
            sns.despine(ax=this_ax)
        if show_legend:
            verbal_legend_lines = [dummy_legend_handle] + [
                Line2D([0], [0], color=value, lw=1)
                for key, value in audible_hue_map.items()
                ]
            verbal_ax[1].legend(
                verbal_legend_lines, ["Report", '"Left"', '"Right"'],
                loc='center left', bbox_to_anchor=(1.025, .5), borderaxespad=0.)

        verbal_ax_trans = transforms.blended_transform_factory(verbal_ax[1].transAxes, verbal_ax[1].transData)
        verbal_ax[1].text(
            side_label_offset, -0.33, "Left",
            transform=verbal_ax_trans, color=audible_hue_map['Left Foot'], **side_label_opts)
        verbal_ax[1].text(
            side_label_offset, -0.66, "Right",
            transform=verbal_ax_trans, color=audible_hue_map['Right Foot'], **side_label_opts)

    stim_ax_trans = transforms.blended_transform_factory(stim_ax[1].transAxes, stim_ax[1].transData)
    raster_linewidth = .5
    mixin_color = np.asarray([1., 1., 1.])
    max_mixin = 0.5
    skip_every = 10
    if show_legend:
        all_custom_stim_legend_lines = []
    electrode_labels_text_ypos = []

    for line_idx, config_str in enumerate(electrode_labels):
        left_mask_spikes = (plot_stim_spikes['timestamp_usec'] == align_timestamps[0]) & (plot_stim_spikes['elecConfig_str'] == config_str)
        right_mask_spikes = (plot_stim_spikes['timestamp_usec'] == align_timestamps[1]) & (plot_stim_spikes['elecConfig_str'] == config_str)
        if left_mask_spikes.any() or right_mask_spikes.any():
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
            if show_legend:
                all_custom_stim_legend_lines.append(GradientLine(cmap))
            stim_ax[1].text(
                side_label_offset, - 0.5 - line_idx, pretty_electrode_labels[line_idx],
                transform=stim_ax_trans, color=electrode_hue_map[config_str], **side_label_opts)

    stim_ax[0].set_xlim([left_sweep * 1e-6, left_sweep * 1e-6 + baseline_dur])
    stim_ax[0].set_ylim([- line_idx - 1.5, .5])
    stim_ax[1].set_xlim([left_sweep * 1e-6, right_sweep * 1e-6])
    stim_ax[1].set_ylim([- line_idx - 1.5, .5])

    '''all_custom_stim_legend_lines = [
        Line2D([0], [0], color=electrode_hue_map[nm], lw=1)
        for idx, nm in enumerate(electrode_labels)
    ]'''
    if show_legend:
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
        this_ax.set_xlabel('')
        sns.despine(ax=this_ax)

    stim_ax[1].set_ylabel('')
    stim_ax[0].set_ylabel('Stim. pulses')
    ##
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
        x='time_sec', y='signal', lw=1,
        # hue='pretty_label', lw=1,
        hue='pretty_label', palette=kin_hue_map, legend=False
    )
    sns.lineplot(
        data=plot_kin.loc[right_mask_kin, :], ax=kin_ax[1],
        x='time_sec', y='signal', lw=1,
        # hue='pretty_label', lw=1,
        hue='pretty_label', palette=kin_hue_map, legend=False
    )

    present_points = plot_kin.loc[plot_kin['pretty_label'].isin(pretty_control_labels), 'pretty_label'].unique()
    if show_legend:
        custom_legend_lines = [dummy_legend_handle] + [
            Line2D([0], [0], color=kin_hue_map[lbl], lw=1)
            for idx, lbl in enumerate(present_points)
        ]
        kin_ax[1].legend(
            custom_legend_lines, ['Joint'] + present_points.tolist(),
            loc='center left', bbox_to_anchor=(1.025, .5), borderaxespad=0.)

    for this_ax in kin_ax:
        this_ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        if this_ax not in bottom_ax:
            plt.setp(this_ax.get_xticklabels(), visible=False)
        this_ax.set_xlabel('')
        sns.despine(ax=this_ax)

    kin_ax[0].xaxis.set_major_locator(ticker.FixedLocator(time_ticks_left))
    kin_ax[0].xaxis.set_minor_locator(ticker.FixedLocator(np.arange(left_sweep * 1e-6, left_sweep * 1e-6 + baseline_dur, 1)))
    kin_ax[1].xaxis.set_major_locator(ticker.FixedLocator(time_ticks_right))
    kin_ax[1].xaxis.set_minor_locator(ticker.FixedLocator(np.arange(left_sweep * 1e-6, right_sweep * 1e-6, 1)))
    kin_ax[1].set_yticklabels([])
    kin_ax[1].set_ylabel('')
    kin_ax[0].set_ylabel('Joint angle\n(deg.)')

    kin_ax_trans = transforms.blended_transform_factory(kin_ax[1].transAxes, kin_ax[1].transData)
    for pretty_kin_label, group in plot_kin.loc[right_mask_kin, :].groupby('pretty_label'):
        kin_ax[1].text(
            side_label_offset, group['signal'].mean(), pretty_kin_label,
            transform=kin_ax_trans, color=kin_hue_map[pretty_kin_label], **side_label_opts)
    #######################

    left_mask_kin = (plot_kin['timestamp_usec'] == align_timestamps[0]) & (plot_kin['pretty_label'].isin(pretty_outcome_labels))
    right_mask_kin = (plot_kin['timestamp_usec'] == align_timestamps[1]) & (plot_kin['pretty_label'].isin(pretty_outcome_labels))
    sns.lineplot(
        data=plot_kin.loc[left_mask_kin, :], ax=outcome_ax[0],
        x='time_sec', y='signal', lw=1,
        # hue='pretty_label', lw=1,
        hue='pretty_label', palette=kin_hue_map, legend=False
    )
    sns.lineplot(
        data=plot_kin.loc[right_mask_kin, :], ax=outcome_ax[1],
        x='time_sec', y='signal', lw=1,
        # hue='pretty_label', lw=1,
        hue='pretty_label', palette=kin_hue_map, legend=False
    )

    present_points = plot_kin.loc[plot_kin['pretty_label'].isin(pretty_outcome_labels), 'pretty_label'].unique()
    if show_legend:
        custom_legend_lines = [dummy_legend_handle] + [
            Line2D([0], [0], color=kin_hue_map[lbl], lw=1)
            for idx, lbl in enumerate(present_points)
        ]
        outcome_ax[1].legend(
            custom_legend_lines, ['Joint'] + present_points.tolist(),
            loc='center left', bbox_to_anchor=(1.025, .5), borderaxespad=0.)

    outcome_ax_trans = transforms.blended_transform_factory(outcome_ax[1].transAxes, outcome_ax[1].transData)
    for pretty_kin_label, group in plot_kin.loc[right_mask_kin, :].groupby('pretty_label'):
        outcome_ax[1].text(
            side_label_offset, group['signal'].mean(), pretty_kin_label,
            transform=outcome_ax_trans, color=kin_hue_map[pretty_kin_label], **side_label_opts)

    for this_ax in outcome_ax:
        this_ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        if this_ax not in bottom_ax:
            plt.setp(this_ax.get_xticklabels(), visible=False)
        this_ax.set_xlabel('')
        sns.despine(ax=this_ax)

    outcome_ax[0].xaxis.set_major_locator(ticker.FixedLocator(time_ticks_left))
    outcome_ax[0].xaxis.set_minor_locator(ticker.FixedLocator(np.arange(left_sweep * 1e-6, left_sweep * 1e-6 + baseline_dur, 1)))
    outcome_ax[1].xaxis.set_major_locator(ticker.FixedLocator(time_ticks_right))
    outcome_ax[1].xaxis.set_minor_locator(ticker.FixedLocator(np.arange(left_sweep * 1e-6, right_sweep * 1e-6, 1)))

    outcome_ax[1].set_yticklabels([])
    outcome_ax[1].set_ylabel('')
    if which_outcome == 'displacement':
        outcome_ax[0].set_ylabel('Marker height\n(mm)')
    elif which_outcome == 'angle':
        outcome_ax[0].set_ylabel('joint angle\n(deg.)')

    if has_audible_timing:
        verbal_ax[0].set_ylabel('Verbal\nreport')
        for this_ax in verbal_ax:
            this_ax.set_yticklabels([])
        verbal_ax[0].xaxis.set_major_locator(ticker.FixedLocator(time_ticks_left))
        verbal_ax[0].xaxis.set_minor_locator(ticker.FixedLocator(np.arange(left_sweep * 1e-6, right_sweep * 1e-6, 1)))
        verbal_ax[1].xaxis.set_major_locator(ticker.FixedLocator(time_ticks_right))
        verbal_ax[1].xaxis.set_minor_locator(ticker.FixedLocator(np.arange(left_sweep * 1e-6, right_sweep * 1e-6, 1)))

    for this_ax in bottom_ax:
        this_ax.set_xlabel('Time (sec.)')
        plt.setp(this_ax.get_xticklabels(), visible=True)

    bottom_ax[0].xaxis.set_major_locator(ticker.FixedLocator(time_ticks_left))
    bottom_ax[0].xaxis.set_minor_locator(ticker.FixedLocator(np.arange(left_sweep * 1e-6, left_sweep * 1e-6 + baseline_dur, 1)))
    bottom_ax[1].xaxis.set_major_locator(ticker.FixedLocator(time_ticks_right))
    bottom_ax[1].xaxis.set_minor_locator(ticker.FixedLocator(np.arange(left_sweep * 1e-6, right_sweep * 1e-6, 1)))


    if color_background:
        these_lims = vspan_limits[0]
        for train_idx, lims in enumerate(these_lims):
            emg_ax[1].axvspan(*lims, facecolor=vspan_colors[0][train_idx], alpha=vspan_alpha)
            kin_ax[1].axvspan(*lims, facecolor=vspan_colors[0][train_idx], alpha=vspan_alpha)
            stim_ax[1].axvspan(*lims, facecolor=vspan_colors[0][train_idx], alpha=vspan_alpha)
            outcome_ax[1].axvspan(*lims, facecolor=vspan_colors[0][train_idx], alpha=vspan_alpha)
            if has_audible_timing:
                verbal_ax[1].axvspan(*lims, facecolor=vspan_colors[0][train_idx], alpha=vspan_alpha)

    top_ax[0].set_title('Baseline')
    top_ax[1].set_title(condition_name)
    fig.align_labels()
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()
