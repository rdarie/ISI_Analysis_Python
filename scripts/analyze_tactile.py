
import traceback
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from isicpy.lookup_tables import emg_montages, muscle_names
from pathlib import Path
import pandas as pd
import numpy as np
import cloudpickle as pickle
import pdb
import gc
from tqdm import tqdm
from scipy import signal
import os

import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    mpl.use('Agg')   # generate postscript output
else:
    mpl.use('QT5Agg')   # generate interactive output

useDPI = 72
dpiFactor = 72 / useDPI
font_zoom_factor = 1.
snsRCParams = {
    'figure.dpi': useDPI, 'savefig.dpi': useDPI,
    'lines.linewidth': .5,
    'lines.markersize': 2.,
    'patch.linewidth': .5,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.linewidth": .25,
    "grid.linewidth": .2,
    "font.size": 5 * font_zoom_factor,
    "axes.axisbelow": False,
    "axes.labelpad": -1 * font_zoom_factor,
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
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "xtick.direction": 'in',
    "ytick.direction": 'in',
}
mplRCParams = {
    'figure.titlesize': 6 * font_zoom_factor,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}
sns.set(
    context='paper', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True, rc=snsRCParams
)
for rcK, rcV in mplRCParams.items():
    mpl.rcParams[rcK] = rcV

left_sweep = int(-0.005 * 1e6)
right_sweep = int(0.015 * 1e6)
probe_channel_name = "analog 2"
verbose = 1

filterOpts = {
    'high': {
        'Wn': 60.,
        'N': 4,
        'btype': 'high',
        'ftype': 'butter'
    },
}

folder_name = "Day8_PM"
blocks_list = [2]
this_emg_montage = emg_montages['lower_v2']
locations_files = {
    'Day8_PM':
        {
            2: '/users/rdarie/Desktop/ISI-C-003/6_Video/Day8_PM_Block0002_tactile_locations.csv'}
}

blocks_list_str = '_'.join(f"{block_idx}" for block_idx in blocks_list)
data_path = Path(f"/users/rdarie/Desktop/ISI-C-003/3_Preprocessed_Data/{folder_name}")
parquet_folder = data_path / "parquets"

lfp_sample_rate = int(1.5e4)
nominal_dt = float(lfp_sample_rate) ** -1 * 1e6
epoch_t = np.arange(left_sweep, right_sweep, nominal_dt)
nominal_num_samp = epoch_t.shape[0]


def assign_locations(t, loc_df=None):
    for row_idx, row in loc_df.iterrows():
        if (t > row['t_start']) & (t < row['t_end']):
            return row['side'], row['location']
    return None, None

all_aligned_lfp = {}
for block_idx in blocks_list:
    locations_df = pd.read_csv(locations_files[folder_name][block_idx])

    # emg_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_emg_df.parquet"
    # this_emg = pd.read_parquet(emg_parquet_path)

    nf7_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_nf7_df.parquet"
    this_lfp = pd.read_parquet(nf7_parquet_path)
    analog_time_vector = np.asarray(this_lfp.index)

    if False:
        filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), lfp_sample_rate)
        this_lfp = pd.DataFrame(
            signal.sosfiltfilt(filterCoeffs, this_lfp - this_lfp.mean(), axis=0),
            index=this_lfp.index, columns=this_lfp.columns)

    ns5_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_ns5_df.parquet"
    this_ns5 = pd.read_parquet(ns5_parquet_path)

    signal_bounds = this_ns5[probe_channel_name].quantile([1e-2, 1-1e-2])
    signal_thresh = (signal_bounds.iloc[-1] + signal_bounds.iloc[0]) / 2
    align_timestamps, cross_mask = getThresholdCrossings(
        this_ns5[probe_channel_name], thresh=signal_thresh, fs=1.5e4)
    if False:
        fig, ax = plt.subplots()
        ax.plot(align_timestamps, align_timestamps ** 0, 'r*')
        ax.plot(this_ns5.index, this_ns5[probe_channel_name])
        plt.show()
    align_timestamps_metadata = pd.DataFrame(
        (pd.Series(align_timestamps) * 1e-6).apply(lambda x: assign_locations(x, loc_df=locations_df)).to_list(),
        index=align_timestamps, columns=['side', 'location'])
    print(f'Epoching lfp from \n\t{nf7_parquet_path}')
    aligned_dfs = {}

    for timestamp in tqdm(align_timestamps.to_numpy()):
        this_side, this_loc = align_timestamps_metadata.loc[timestamp, ['side', 'location']]
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
            this_mask = (
                    (analog_time_vector >= timestamp + left_sweep - nominal_dt / 2) &
                    (analog_time_vector < timestamp + right_sweep + sweep_offset + nominal_dt / 2))
            if verbose > 1:
                print(f'this_mask.sum() = {this_mask.sum()}')
        aligned_dfs[(timestamp, this_side, this_loc)] = pd.DataFrame(
            this_lfp.loc[this_mask, :].to_numpy(), index=epoch_t, columns=this_lfp.columns)
        aligned_dfs[(timestamp, this_side, this_loc)] = aligned_dfs[(timestamp, this_side, this_loc)] - aligned_dfs[(timestamp, this_side, this_loc)].mean()
    all_aligned_lfp[block_idx] = pd.concat(aligned_dfs, names=['timestamp_usec', 'side', 'location', 'time_usec'])

lfp_df = pd.concat(all_aligned_lfp, names=['block', 'timestamp_usec', 'side', 'location', 'time_usec'])
lfp_df.columns.name = 'channel'

aucs_df = (lfp_df ** 2).groupby(['block', 'timestamp_usec']).mean() ** 0.5
scaler = StandardScaler().fit(aucs_df)
standard_aucs_df = pd.DataFrame(
    scaler.transform(aucs_df), index=aucs_df.index,
    columns=aucs_df.columns
)
'''
scaler = StandardScaler().fit(aucs_df.to_numpy().reshape(-1, 1))
standard_aucs_df = aucs_df.copy().apply(lambda x: scaler.transform(x.to_numpy().reshape(-1, 1)).flatten(), axis='columns', result_type='broadcast')
'''

'''
ax = sns.histplot(standard_aucs_df.stack())
plt.show()
'''

outlier_factor = 3
outlier_lookup = (standard_aucs_df > outlier_factor).stack()

show_plots = False
pdf_folder = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/5_Figures/{folder_name}")
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)

# list_of_plot_types = ['mean', 'median', 'singles']
list_of_plot_types = ['mean', 'median']
for plot_type in list_of_plot_types:
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_lfp_tactile_response_{plot_type}.pdf")
    if plot_type == 'median':
        relplot_kwargs = dict(estimator=np.median, errorbar='se',)
    if plot_type == 'mean':
        relplot_kwargs = dict(estimator=np.mean, errorbar='se',)
    elif plot_type == 'singles':
        relplot_kwargs = dict(estimator=None, units='timestamp_usec', alpha=.5)
    with PdfPages(pdf_path) as pdf:
        plot_df = lfp_df.stack().to_frame(name='value').reset_index()
        # is_outlier = pd.MultiIndex.from_frame(plot_df.loc[:, ['block', 'timestamp_usec', 'channel']]).map(outlier_lookup).to_numpy()
        # plot_df = plot_df.loc[~is_outlier, :]
        # plot_df.sort_values(['block', 'timestamp_usec', 'channel'], inplace=True)
        plot_df.loc[:, 'is_outlier'] = pd.MultiIndex.from_frame(plot_df.loc[:, ['block', 'timestamp_usec', 'channel']]).map(outlier_lookup).to_numpy()
        plot_df.loc[:, 'time_sec'] = plot_df['time_usec'] * 1e-6
        for name, group in tqdm(plot_df.groupby(['location'])):
            g = sns.relplot(
                data=group.loc[~group['is_outlier'], :],
                x='time_sec', y='value',
                col='channel', col_wrap=8,
                hue='side',
                kind='line',
                facet_kws=dict(sharey=False),
                height=1.5, aspect=1.5,
                **relplot_kwargs
            )
            for ax in g.axes.flatten():
                ax.axvline(0, color='r')
            g.set_titles(col_template='{col_name}')
            g.figure.suptitle(f"{name}", fontsize=10)
            g.figure.tight_layout()
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if show_plots:
                plt.show()
            else:
                plt.close()
