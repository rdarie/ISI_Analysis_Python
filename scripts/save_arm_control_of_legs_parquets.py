
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

plots_dt = int(1e3)  #  usec

folder_name = "Day7_AM"
block_idx_list = [4]
this_emg_montage = emg_montages['lower']

# folder_name = "Day8_AM"
# block_idx_list = [1, 2, 3, 4]
# this_emg_montage = emg_montages['lower']

# folder_name = "Day11_AM"
# block_idx_list = [1, 2, 4]
# this_emg_montage = emg_montages['lower_v2']

# folder_name = "Day12_PM"
# block_idx_list = [3, 4]
# this_emg_montage = emg_montages['lower_v2']

for block_idx in block_idx_list:
    if folder_name in ['Day11_AM']:
        angles_dict = {
            'LeftElbow': ['LeftForeArm', 'LeftElbow', 'LeftUpperArm'],
            'RightElbow': ['RightForeArm', 'RightElbow', 'RightUpperArm'],
            'LeftKnee': ['LeftLowerLeg', 'LeftKnee', 'LeftUpperLeg'],
            'RightKnee': ['RightLowerLeg', 'RightKnee', 'RightUpperLeg'],
            }
    elif folder_name in ["Day7_AM", 'Day8_AM', 'Day12_PM']:
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
    standardize_emg = False

    file_path = data_path / f"Block{block_idx:0>4d}_Synced_Session_Data.mat"
    try:
        this_kin_offset = kinematics_offsets[folder_name][block_idx]
    except:
        this_kin_offset = 0

    all_electrodes_across_experiments = [
        "-(14,)+(6, 22)", "-(3,)+(2,)", "-(27,)+(19,)",
        "-(27,)+(26,)", "-(139,)+(131,)", "-(136,)+(144,)",
        "-(131,)+(130,)", "-(155,)+(154,)"
        ]

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

    def adjust_channel_name(cn):
        signal_type, ch_num_str = cn.split(' ')
        elec = int(ch_num_str)
        if elec < 128:
            return f"ch {elec} (caudal)"
        else:
            return f"ch {elec - 128} (rostral)"

    parquet_folder = data_path / "parquets"
    timecode_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_ripple_timecode_df.parquet"
    nf7_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_nf7_df.parquet"
    ns5_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_ns5_df.parquet"
    emg_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_emg_df.parquet"
    stim_info_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_stim_info_df.parquet"
    stim_info_traces_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_stim_info_traces_df.parquet"
    nev_spikes_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_nev_spikes_df.parquet"
    points_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_points_df.parquet"
    audible_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_audible_timings_df.parquet"

    reprocess_raw = True
    save_parquets = True

    if (not os.path.exists(parquet_folder)) or reprocess_raw:
        data_dict = load_synced_mat(
            file_path,
            load_stim_info=True, split_trains=False, stim_info_traces=True, force_trains=False,
            load_ripple=True, ripple_as_df=True, ripple_variable_names=['NEV', 'NS5', 'NF7', 'TimeCode'],  # 'NS5', 'NF7', 'TimeCode'
            load_vicon=True, vicon_as_df=True, interpolate_emg=True, kinematics_time_offset=this_kin_offset,
            vicon_variable_names=['EMG', 'Points'],  # 'Points'
            load_all_logs=False, verbose=1
            )

        ns5_df = data_dict['ripple']['NS5']
        timecode_df = data_dict['ripple']['TimeCode']
        stim_info_trace_df = pd.concat(data_dict['stim_info_traces'], names=['feature'], axis='columns')

        lfp_df = data_dict['ripple']['NF7']
        lfp_df.columns = [adjust_channel_name(cn) for cn in lfp_df.columns]

        emg_df = data_dict['vicon']['EMG'].copy()
        emg_df.rename(columns=this_emg_montage, inplace=True)
        emg_df.index.name = 'time_usec'
        emg_df.drop(['NA'], axis='columns', inplace=True)
        if standardize_emg:
            emg_scaler_path = data_path / "pickles" / "emg_scaler.p"
            with open(emg_scaler_path, 'rb') as handle:
                scaler = pickle.load(handle)
            emg_df.loc[:, :] = scaler.transform(emg_df)

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

        displacements_list = [
            'LeftToe', 'RightToe'
            ]
        for point_name in displacements_list:
            try:
                points_df.loc[:, (point_name, 'displacement')] = (np.sqrt(
                    points_df.loc[:, (point_name, 'x')].to_numpy() ** 2 +
                    points_df.loc[:, (point_name, 'y')].to_numpy() ** 2 +
                    points_df.loc[:, (point_name, 'z')].to_numpy() ** 2
                ))
            except Exception:
                print(f'\nSkipping displacement calc for {point_name}\n')
                traceback.print_exc()

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
            lfp_df.to_parquet(nf7_parquet_path)
            ns5_df.to_parquet(ns5_parquet_path)
            stim_info_trace_df.to_parquet(stim_info_traces_parquet_path)
            timecode_df.to_parquet(timecode_parquet_path)
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
