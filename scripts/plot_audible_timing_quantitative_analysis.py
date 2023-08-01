
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
from isicpy.utils import makeFilterCoeffsSOS, closestSeries, timestring_to_timestamp
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
        "xtick.bottom": False,
        "xtick.top": False,
        "ytick.left": True,
        "ytick.right": False,
        "xtick.major.width": .25,
        "ytick.major.width": .25,
        "xtick.minor.width": .25,
        "ytick.minor.width": .25,
        "xtick.major.size": 1.5,
        "ytick.major.size": 1.5,
        "xtick.minor.size": .5,
        "ytick.minor.size": .5,
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

plots_dt = int(1e3)  #  usec

folder_name = "Day11_AM"
block_idx_list = [4]
this_emg_montage = emg_montages['lower_v2']

if folder_name in ['Day11_AM']:
    angles_dict = {
        'LeftElbow': ['LeftForeArm', 'LeftElbow', 'LeftUpperArm'],
        'RightElbow': ['RightForeArm', 'RightElbow', 'RightUpperArm'],
        'LeftKnee': ['LeftLowerLeg', 'LeftKnee', 'LeftUpperLeg'],
        'RightKnee': ['RightLowerLeg', 'RightKnee', 'RightUpperLeg'],
        }

data_path = Path(f"/users/rdarie/Desktop/ISI-C-003/3_Preprocessed_Data/{folder_name}")
pdf_folder = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/5_Figures/{folder_name}")

if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)

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


def int_or_nan(x):
    try:
        return int(x)
    except:
        return np.nan

all_electrodes_across_experiments = [
    "-(14,)+(6, 22)", "-(3,)+(2,)", "-(27,)+(19,)",
    "-(27,)+(26,)", "-(139,)+(131,)", "-(136,)+(144,)",
    "-(131,)+(130,)", "-(155,)+(154,)"]

if folder_name in ['Day11_AM']:
    baseline_timestamps = [int(179e6) + 1500, int(179e6) + 1500]
    align_timestamps = [baseline_timestamps[0], 500046533]

    only_these_electrodes = [
        "-(14,)+(6, 22)", "-(3,)+(2,)", "-(27,)+(19,)", "-(27,)+(26,)", "-(131,)+(130,)", "-(155,)+(154,)"]
    stim_ax_height = 0.5 * len(only_these_electrodes)
    # emg_ax_height = 14 - stim_ax_height

    fps = 29.97
    audible_timing_path = Path(
        f"/users/rdarie/Desktop/ISI-C-003/6_Video/Day11_AM_Cam1_GH010630_NDF_audible_timings_sanitized_rd.csv")
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

def parse_json_colors(x):
    if x['palette_type'] == 'palettable':
        module_str = 'palettable.' + x['palettable_module']
        palette_module = eval(module_str)
        return palette_module.mpl_colors[x['palette_idx']]
    elif x['palette_type'] == 'hex':
        return mpl.colors.to_rgb(x['palette_idx'])


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
elec_format_df.loc[:, 'pretty_label_w_array'] = elec_format_df.apply(lambda x: f"{x['pretty_label']}\n({x['which_array']})", axis='columns')


reporting_time_ranges = {
    2: [
        [349, 371], [418, 441], [758, 845],
    ],
    4: [
        [256, 265], [322.75, 367.56], [412.5, 421.], [462.5, 558.9],
        [578, 596, ]
    ]
}

nev_train_dict = {}
block_offset = 0.
for block_idx in block_idx_list:

    if standardize_emg:
        emg_scaler_path = data_path / "pickles" / "emg_scaler.p"
        with open(emg_scaler_path, 'rb') as handle:
            scaler = pickle.load(handle)

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

    audible_timing = pd.read_csv(audible_timing_path)
    audible_timing.loc[:, 'timestamp'] = (
        audible_timing
        .apply(
            lambda x: timestring_to_timestamp(x['time'], day=31, fps=fps, timecode_type='NDF'),
            axis='columns'))
    first_audible_ts, last_audible_ts = audible_timing['timestamp'].iloc[0], audible_timing['timestamp'].iloc[-1]
    if last_audible_ts < first_audible_ts:
        # rollover
        audible_timing.loc[audible_timing['timestamp'] < first_audible_ts, 'timestamp'] += pd.Timedelta(24, unit='hour')
    timecode_df = pd.read_parquet(ripple_timecode_parquet_path)
    timecode_df.loc[:, 'timestamp'] = (
        timecode_df
        .apply(
            lambda x: timestring_to_timestamp(x['TimeString'], day=31, fps=fps, timecode_type='NDF'),
            axis='columns'))
    first_timecode_ts, last_timecode_ts = timecode_df['timestamp'].iloc[0], timecode_df['timestamp'].iloc[-1]
    if last_timecode_ts < first_timecode_ts:
        # rollover
        timecode_df.loc[timecode_df['timestamp'] < first_timecode_ts, 'timestamp'] += pd.Timedelta(24, unit='hour')
    if block_idx == 4:
        timecode_df.loc[:, 'timestamp'] += pd.Timedelta(24, unit='hour')
    audible_timing.loc[:, 'ripple_time'] = (audible_timing['timestamp'] - timecode_df['timestamp'].iloc[0]).apply(
        lambda x: x.total_seconds()) + timecode_df['PacketTime'].iloc[0]
    is_in_block = (audible_timing['ripple_time'] >= emg_df.index[0] * 1e-6) & (audible_timing['ripple_time'] <= emg_df.index[-1] * 1e-6)
    audible_timing = audible_timing.loc[is_in_block, :]

    rostral_nev_spikes = nev_spikes.loc[nev_spikes['Electrode'] > 128, :].drop_duplicates('TimeStamp')

    train_thresh = .25
    elec_trains = {}
    for name, group in rostral_nev_spikes.groupby('elecConfig_str'):
        print(f'{name}:')
        time_since = group['time_seconds'].diff()
        time_since.iloc[0] = np.inf
        these_first_timestamps = group.loc[time_since > train_thresh, 'time_seconds']
        time_after = group['time_seconds'].diff(periods=-1) * -1
        time_after.iloc[-1] = np.inf
        these_last_timestamps = group.loc[time_after > train_thresh, 'time_seconds']
        #
        include_these = pd.Series(False, index=these_first_timestamps.index)
        for t_start, t_stop in reporting_time_ranges[block_idx]:
            include_these = include_these | ((these_first_timestamps >= t_start) & (these_first_timestamps <= t_stop))
        elec_trains[name] = rostral_nev_spikes.loc[include_these.index[include_these], :]

        closestTimes, closestIdx = closestSeries(elec_trains[name]['time_seconds'], these_last_timestamps, strictly='greater')
        elec_trains[name].loc[:, 'train_end_time'] = closestTimes.to_numpy()
        elec_trains[name].loc[:, 'train_duration'] = elec_trains[name]['train_end_time'] - elec_trains[name]['time_seconds']

    nev_train_dict[block_idx] = pd.concat(elec_trains, ignore_index=True).sort_values('TimeStamp')
    nev_train_dict[block_idx].loc[:, 'report'] = None
    nev_train_dict[block_idx].loc[:, 'report_latency'] = None
    for idx, (name, this_train) in enumerate(nev_train_dict[block_idx].iterrows()):
        if idx < nev_train_dict[block_idx].shape[0] - 1:
            next_train = nev_train_dict[block_idx].iloc[idx + 1, :]
            valid_reports_mask = (audible_timing['ripple_time'] >= this_train['time_seconds']) & (audible_timing['ripple_time'] <= next_train['time_seconds'])
        else:
            valid_reports_mask = (audible_timing['ripple_time'] >= this_train['time_seconds'])
        if valid_reports_mask.any():
            this_report = audible_timing.loc[valid_reports_mask, :].iloc[0, :]
            nev_train_dict[block_idx].loc[name, 'report'] = this_report['words']
            nev_train_dict[block_idx].loc[name, 'report_time'] = this_report['ripple_time']
            nev_train_dict[block_idx].loc[name, 'report_latency'] = this_report['ripple_time'] - this_train['time_seconds']
        else:
            nev_train_dict[block_idx].loc[name, 'report'] = 'NA'
            nev_train_dict[block_idx].loc[name, 'report_latency'] = -1
            nev_train_dict[block_idx].loc[name, 'report_time'] = -1
    nev_train_dict[block_idx].loc[:, 'time_seconds'] += block_offset
    nev_train_dict[block_idx].loc[:, 'train_end_time'] += block_offset
    nev_train_dict[block_idx].loc[:, 'report_time'] += block_offset
    nev_train_dict[block_idx].loc[:, 'time_usec'] += int(block_offset * 1e6)
    block_offset += emg_df.index[-1] * 1e-6

nev_trains = pd.concat(nev_train_dict, names=['block']).reset_index().sort_values(['block', 'TimeStamp'])

correct_responses = {
    '-(131,)+(130,)': 'Left',
    '-(155,)+(154,)': 'Right'
}

nev_trains.loc[:, 'report_correct'] = nev_trains.apply(lambda x: x['report'] == correct_responses[x['elecConfig_str']], axis='columns')
nev_trains.loc[:, 'report_type'] = ''
nev_trains.loc[nev_trains['report'] == 'NA', 'report_type'] = 'No report'
nev_trains.loc[(nev_trains['report'] != 'NA') & nev_trains['report_correct'], 'report_type'] = 'Correct'
nev_trains.loc[(nev_trains['report'] != 'NA') & ~nev_trains['report_correct'], 'report_type'] = 'Incorrect'

display_order = ['Correct', 'Incorrect', 'No report']
sort_key = lambda x: display_order.index(x)
nev_trains.loc[:, 'display_order'] = nev_trains['report_type'].apply(sort_key)
nev_trains.sort_values('display_order', inplace=True)

nev_trains.loc[:, 'elecConfig_pretty'] = nev_trains['elecConfig_str'].map(elec_format_df.set_index('label')['pretty_label_w_array'])

counts_dict = {}
for name, group in nev_trains.groupby('elecConfig_pretty'):
    counts_dict[name] = group.groupby('report_type').count().iloc[:, 0].to_frame(name='count')
    counts_dict[name].loc[:, 'proportion'] = counts_dict[name]['count'] / counts_dict[name]['count'].sum()


fig, ax = plt.subplots()
ax.scatter(
    nev_trains.loc[nev_trains['report_time'] > 0, 'report_time'],
    nev_trains.loc[nev_trains['report_time'] > 0, 'report_time'] ** 0,
    marker='d', label='report')
for name, group in nev_trains.groupby('elecConfig_str'):
    ax.scatter(group['time_seconds'], group['time_seconds'] ** 0, label=name)
for name, row in nev_trains.iterrows():
    ax.text(row['time_seconds'], 1., row['report'])
ax.legend()

block_list_string = '_'.join([f"{bidx}" for bidx in block_idx_list])
pdf_path = pdf_folder / Path(f"{folder_name}_Blocks_{block_list_string}_treadmill_audible_responses.pdf")
show_plots = True
with PdfPages(pdf_path) as pdf:
    w, h = 1., 1.2
    palette = {
        'No report': '#BBBBBB',
        'Correct': '#009988',
        'Incorrect': '#CC3311'
        }

    g = sns.displot(
        data=nev_trains,
        col='elecConfig_pretty',
        x='report_type',
        hue='report_type', palette=palette,
        height=h, aspect=w / h,
        # multiple='dodge', shrink=0.9,
        )

    for elec, ax in g.axes_dict.items():
        for idx, (row_idx, row) in enumerate(counts_dict[elec].iterrows()):
            ax.text(idx, 1, f"{int(row['proportion'] * 100)} %", ha='center')
    g.set_titles(col_template='{col_name}')
    g.set_xlabels('')
    g.set_ylabels('Number of steps')
    g._legend.set_title('Response type')
    g.figure.suptitle('Proportion correct responses')
    g.figure.tight_layout()
    g.figure.align_labels()
    g.figure.set_size_inches(2.6, 1.2)
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()

    w, h = 1., 1.2
    print('response latencies (sec.)')
    for name, group in nev_trains.loc[nev_trains['report_latency'] > 0, :].groupby(['elecConfig_pretty', 'report_type']):
        print(f"{name[0]} ({name[1]}):")
        print(f"\t mean: {group['report_latency'].mean():.2f}")
        print(f"\t std: {group['report_latency'].std():.2f}")
    g = sns.displot(
        data=nev_trains.query('report_latency > 0'),
        col='elecConfig_pretty', x='report_latency', hue='report_type',
        palette=palette,
        height=h, aspect=w / h, legend=False,
        )
    g.set_titles(col_template='{col_name}')
    g.set_xlabels('Response latency (sec.)')
    g.set_ylabels('Number of steps')
    g.figure.suptitle('Response latency')
    g.figure.tight_layout()
    g.figure.align_labels()
    g.figure.set_size_inches(2.6, 1.2)
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()

    breaktimes = [
        [6, 62],
        [109, 154],
        [161, 200],
        [301, 319]
        ]
    nev_trains.sort_values('time_seconds', inplace=True)
    nev_trains.index = pd.TimedeltaIndex(nev_trains['time_seconds'], unit='s')

    def extract_proportion_correct(df):
        data = elec_group.loc[df.index, 'report_type']
        return (data == 'Correct').sum() / data.shape[0]
    def extract_proportion_incorrect(df):
        data = elec_group.loc[df.index, 'report_type']
        return (data == 'Incorrect').sum() / data.shape[0]
    def extract_proportion_none(df):
        data = elec_group.loc[df.index, 'report_type']
        return (data == 'No report').sum() / data.shape[0]


    window_width = pd.Timedelta(2, unit='min')

    prop_correct_dict = {}
    prop_incorrect_dict = {}
    prop_none_dict = {}

    for elecConfig, elec_group in nev_trains.groupby('elecConfig_str'):
        prop_correct_dict[elecConfig] = elec_group['block'].rolling(window=window_width, center=True).apply(extract_proportion_correct)
        prop_correct_dict[elecConfig].index = prop_correct_dict[elecConfig].index.total_seconds()
        prop_incorrect_dict[elecConfig] = elec_group['block'].rolling(window=window_width, center=True).apply(extract_proportion_incorrect)
        prop_incorrect_dict[elecConfig].index = prop_incorrect_dict[elecConfig].index.total_seconds()
        prop_none_dict[elecConfig] = elec_group['block'].rolling(window=window_width, center=True).apply(extract_proportion_none)
        prop_none_dict[elecConfig].index = prop_none_dict[elecConfig].index.total_seconds()

    prop_correct = pd.concat(prop_correct_dict, names=['elecConfig_str']).reset_index()
    prop_correct.rename(columns={'block': 'proportion'}, inplace=True)
    prop_incorrect = pd.concat(prop_incorrect_dict, names=['elecConfig_str']).reset_index()
    prop_incorrect.rename(columns={'block': 'proportion'}, inplace=True)
    prop_none = pd.concat(prop_none_dict, names=['elecConfig_str']).reset_index()
    prop_none.rename(columns={'block': 'proportion'}, inplace=True)

    all_props = pd.concat({
        'Correct': prop_correct,
        'Incorrect': prop_incorrect,
        'No report': prop_none,
    }, names=['response_type']).reset_index()
    all_props.loc[:, 'elecConfig_pretty'] = all_props['elecConfig_str'].map(elec_format_df.set_index('label')['pretty_label_w_array'])
    all_props.loc[:, 'time_seconds'] -= all_props['time_seconds'].iloc[0]
    w, h = 2, 2
    g = sns.relplot(
        data=all_props, x='time_seconds', y='proportion', hue='response_type',
        col='elecConfig_pretty',
        kind='scatter', palette=palette, height=h, aspect=w / h,
        )
    for break_start, break_stop in breaktimes:
        for this_ax in g.axes.flatten():
            this_ax.axvspan(break_start, break_stop, color='r', alpha=0.1)
    g.set_titles(col_template='{col_name}')
    g.set_xlabels('Time (sec.)')
    g._legend.set_title('Response type')
    g.figure.suptitle('Proportion correct responses')
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()

