
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

folder_name = "Day8_PM"
block_idx_list = [2]
this_emg_montage = emg_montages['lower']

for block_idx in block_idx_list:
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
    # stim_info_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_stim_info_df.parquet"
    # stim_info_traces_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_stim_info_traces_df.parquet"
    # nev_spikes_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_nev_spikes_df.parquet"
    # points_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_points_df.parquet"
    # audible_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_audible_timings_df.parquet"

    save_parquets = True
    data_dict = load_synced_mat(
        file_path,
        load_stim_info=False,
        load_ripple=True, ripple_as_df=True, ripple_variable_names=['NEV', 'NS5', 'NF7', 'TimeCode'],
        load_vicon=True, vicon_as_df=True, interpolate_emg=True, kinematics_time_offset=this_kin_offset,
        vicon_variable_names=['EMG'],  # 'Points'
        load_all_logs=False, verbose=1
        )

    ns5_df = data_dict['ripple']['NS5']
    timecode_df = data_dict['ripple']['TimeCode']

    lfp_df = data_dict['ripple']['NF7']
    lfp_df.columns = [adjust_channel_name(cn) for cn in lfp_df.columns]

    emg_df = data_dict['vicon']['EMG'].copy()
    emg_df.rename(columns=this_emg_montage, inplace=True)
    emg_df.index.name = 'time_usec'
    emg_df.drop(['NA'], axis='columns', inplace=True)

    if save_parquets:
        if not os.path.exists(parquet_folder):
            os.makedirs(parquet_folder)
        emg_df.to_parquet(emg_parquet_path)
        lfp_df.to_parquet(nf7_parquet_path)
        ns5_df.to_parquet(ns5_parquet_path)
        timecode_df.to_parquet(timecode_parquet_path)
