
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

# folder_name = "Day7_AM"
# block_idx_list = [4]
# this_emg_montage = emg_montages['lower']

# folder_name = "Day8_AM"
# block_idx_list = [1, 2, 3, 4]
# this_emg_montage = emg_montages['lower']

# folder_name = "Day8_PM"
# block_idx_list = [2]
# this_emg_montage = emg_montages['lower']

# folder_name = "Day11_AM"
# block_idx_list = [1, 2, 4]
# this_emg_montage = emg_montages['lower_v2']

folder_name = "Day11_PM"
this_emg_montage = emg_montages['lower_v2']
block_idx_list = [5]

# folder_name = "Day12_PM"
# block_idx_list = [3, 4]
# this_emg_montage = emg_montages['lower_v2']

for block_idx in block_idx_list:
    if folder_name in ['Day11_AM', 'Day11PM']:
        angles_dict = {
            'LeftElbow': ['LeftForeArm', 'LeftElbow', 'LeftUpperArm'],
            'RightElbow': ['RightForeArm', 'RightElbow', 'RightUpperArm'],
            'LeftKnee': ['LeftLowerLeg', 'LeftKnee', 'LeftUpperLeg'],
            'RightKnee': ['RightLowerLeg', 'RightKnee', 'RightUpperLeg'],
            }
    elif folder_name in ["Day7_AM", 'Day8_AM', 'Day8_PM', 'Day12_PM']:
        angles_dict = {
            'LeftElbow': ['LeftUpperArm', 'LeftForeArm', 'LeftElbow', ],
            'RightElbow': ['RightUpperArm', 'RightForeArm', 'RightElbow', ],
            'LeftKnee': ['LeftAnkle', 'LeftHip', 'LeftKnee', ],
            'RightKnee': ['RightAnkle', 'RightHip', 'RightKnee', ],
            }

    data_path = Path(f"/users/rdarie/Desktop/ISI-C-003/3_Preprocessed_Data/{folder_name}")

    verbose = 2

    file_path = data_path / f"Block{block_idx:0>4d}_Synced_Session_Data.mat"
    try:
        this_kin_offset = kinematics_offsets[folder_name][block_idx]
    except:
        this_kin_offset = 0

    if folder_name in ['Day7_AM', 'Day8_AM', 'Day8_PM', 'Day11_PM', 'Day12_PM']:
        has_audible_timing = False
    elif folder_name in ['Day11_AM']:
        has_audible_timing = True

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

    save_parquets = True
    load_stim_info = False
    load_nev = False
    load_timecode = False
    data_dict = load_synced_mat(
        file_path,
        load_stim_info=load_stim_info, split_trains=True, force_trains=True, stim_info_traces=load_stim_info,
        load_ripple=True, ripple_as_df=True, ripple_variable_names=['NS5', 'NF7'],  # 'NEV', 'NS5', 'NF7', 'TimeCode'
        load_vicon=True, vicon_as_df=True, interpolate_emg=True, kinematics_time_offset=this_kin_offset,
        vicon_variable_names=['EMG', 'Points'],  # 'Points'
        load_all_logs=False, verbose=1
        )

    ns5_df = data_dict['ripple']['NS5']
    if load_timecode:
        timecode_df = data_dict['ripple']['TimeCode']
    if load_stim_info:
        stim_info_trace_df = pd.concat(data_dict['stim_info_traces'], names=['feature'], axis='columns')

    lfp_df = data_dict['ripple']['NF7']
    lfp_df.columns = [adjust_channel_name(cn) for cn in lfp_df.columns]

    emg_df = data_dict['vicon']['EMG'].copy()
    emg_df.rename(columns=this_emg_montage, inplace=True)
    emg_df.index.name = 'time_usec'
    emg_df.drop(['NA'], axis='columns', inplace=True)

    if 'Points' in data_dict['vicon']:
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
        points_df.to_parquet(points_parquet_path)

    if load_stim_info:
        stim_info_df = data_dict['stim_info']
        stim_info_df.loc[:, 'elecConfig_str'] = stim_info_df.apply(
            lambda x: f'-{x["elecCath"]}+{x["elecAno"]}', axis='columns')
        stim_info_df.loc[:, 'elecAll'] = stim_info_df.apply(lambda x: x['elecCath'] + x['elecAno'], axis='columns')

    if load_nev:
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
        audible_timing.to_parquet(audible_parquet_path)

    if not os.path.exists(parquet_folder):
        os.makedirs(parquet_folder)

    emg_df.to_parquet(emg_parquet_path)
    lfp_df.to_parquet(nf7_parquet_path)
    ns5_df.to_parquet(ns5_parquet_path)

    if load_stim_info:
        stim_info_trace_df.to_parquet(stim_info_traces_parquet_path)
        stim_info_df.to_parquet(stim_info_parquet_path)

    if load_nev:
        nev_spikes.to_parquet(nev_spikes_parquet_path)

    if load_timecode:
        timecode_df.to_parquet(timecode_parquet_path)
