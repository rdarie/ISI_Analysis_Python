
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.use('qtagg')  # generate interactive output
import traceback
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings, mapToDF
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
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler


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
    context='paper', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True, rc=snsRCParams
)
for rcK, rcV in mplRCParams.items():
    mpl.rcParams[rcK] = rcV

left_sweep = int(-0.050 * 1e6)
right_sweep = int(0.125 * 1e6)

verbose = 1

filterOpts = {
    'high': {
        'Wn': 2.,
        'N': 8,
        'btype': 'high',
        'ftype': 'butter'
    },
    'low': {
        'Wn': 250.,
        'N': 8,
        'btype': 'low',
        'ftype': 'butter'
    },
}

folder_name = "Day11_PM"
blocks_list = [5]
this_emg_montage = emg_montages['lower_v2']
artifact_channel_name = 'caudal'
locations_files = {
    'Day11_PM':
        {
            5: '/users/rdarie/Desktop/ISI-C-003/3_Preprocessed_Data/Day11_PM/Day11_PM_Block0005_tens_locations.csv'}
}

blocks_list_str = '_'.join(f"{block_idx}" for block_idx in blocks_list)
data_path = Path(f"/users/rdarie/Desktop/ISI-C-003/3_Preprocessed_Data/{folder_name}")
parquet_folder = data_path / "parquets"

implant_map = mapToDF("/users/rdarie/isi_analysis/ISI_Analysis_Python/ripple_map_files/boston_sci_square.map")
implant_map.loc[:, 'lfp_label'] = implant_map.apply(
    lambda row: f"ch {int(row['elecID'])} ({row['elecName']})", axis='columns')

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

    # nf7_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_reref_lfp_df.parquet"
    nf7_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_nf7_df.parquet"

    this_lfp = pd.read_parquet(nf7_parquet_path)
    analog_time_vector = np.asarray(this_lfp.index)

    downsample_factor = 20
    if True:
        filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), lfp_sample_rate)
        this_lfp = pd.DataFrame(
            signal.sosfiltfilt(filterCoeffs, this_lfp - this_lfp.mean(), axis=0),
            index=this_lfp.index, columns=this_lfp.columns)

    artifact_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_common_average_df.parquet"
    this_artifact = pd.read_parquet(artifact_parquet_path)

    signal_thresh = 2.
    temp = pd.Series(this_artifact[artifact_channel_name].to_numpy())
    _, cross_mask = getThresholdCrossings(
        temp, thresh=signal_thresh, fs=lfp_sample_rate, iti=.5)
    align_timestamps = this_artifact.index[cross_mask].copy()
    # print(pd.Series(align_timestamps).diff())

    if False:
        fig, ax = plt.subplots()
        ax.plot(align_timestamps, align_timestamps ** 0, 'r*')
        ax.plot(this_artifact.index, this_artifact[probe_channel_name])
        plt.show()

    align_timestamps_metadata = pd.DataFrame(
        (pd.Series(align_timestamps) * 1e-6).apply(lambda x: assign_locations(x, loc_df=locations_df)).to_list(),
        index=align_timestamps, columns=['side', 'location'])

    align_timestamps = align_timestamps[~align_timestamps_metadata['side'].isna()]
    align_timestamps_metadata = align_timestamps_metadata[~align_timestamps_metadata['side'].isna()]

    print(f'Epoching lfp from \n\t{nf7_parquet_path}')
    aligned_dfs = {}
    baseline_bounds = [-50e3, 0]
    subtract_baselines = True
    baseline_mask = (epoch_t > baseline_bounds[0]) & (epoch_t < baseline_bounds[1])

    for timestamp in tqdm(align_timestamps.to_numpy()):
        this_side, this_loc = align_timestamps_metadata.loc[timestamp, ['side', 'location']]
        this_mask = (analog_time_vector >= timestamp + left_sweep) & (analog_time_vector <= timestamp + right_sweep)
        first_index = np.flatnonzero(this_mask)[0]
        this_aligned = pd.DataFrame(
            this_lfp.iloc[first_index:first_index + nominal_num_samp, :].to_numpy(),
            index=epoch_t, columns=this_lfp.columns)
        if subtract_baselines:
            baseline = this_aligned.loc[baseline_mask, :].mean()
            aligned_dfs[(timestamp, this_side, this_loc)] = (this_aligned - baseline).iloc[::downsample_factor, :]
        else:
            aligned_dfs[(timestamp, this_side, this_loc)] = this_aligned.iloc[::downsample_factor, :]
    all_aligned_lfp[block_idx] = pd.concat(aligned_dfs, names=['timestamp_usec', 'side', 'location', 'time_usec'])

lfp_df = pd.concat(all_aligned_lfp, names=['block', 'timestamp_usec', 'side', 'location', 'time_usec'])
lfp_df.columns.name = 'channel'

averages_df = lfp_df.groupby(['side', 'location', 'time_usec']).mean()

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

outlier_factor = 1.25
outlier_lookup = (standard_aucs_df > outlier_factor).stack()
print(f"{outlier_lookup.sum() / outlier_lookup.size:.2f} samples discarded as outliers")

# sns.displot(standard_aucs_df.stack())

show_plots = False
pdf_folder = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/5_Figures/{folder_name}")
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)

list_of_plot_types = ['mean', 'singles', 'median']
# list_of_plot_types = ['mean']
for plot_type in list_of_plot_types:
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_lfp_TENS_response_{plot_type}.pdf")
    if plot_type == 'median':
        relplot_kwargs = dict(estimator=np.median, errorbar='se')
    if plot_type == 'mean':
        relplot_kwargs = dict(estimator=np.mean, errorbar='se')
    elif plot_type == 'singles':
        relplot_kwargs = dict(estimator=None, units='timestamp_usec', alpha=.5)

    plot_df = lfp_df.stack().to_frame(name='value').reset_index()

    for name in ['elecName', 'xcoords', 'ycoords']:
        plot_df.loc[:, name] = plot_df['channel'].map(implant_map.set_index('lfp_label')[name])

    plot_df.loc[:, 'ycoords'] = plot_df.loc[:, 'ycoords'] / 1000
    plot_df.loc[:, 'xcoords'] = plot_df.loc[:, 'xcoords'] / 1000

    plot_df.loc[:, 'Poke location / recording paddle'] = plot_df.apply(lambda x: f"{x['side']}, {x['elecName']} array", axis=1)

    is_outlier = pd.MultiIndex.from_frame(plot_df.loc[:, ['block', 'timestamp_usec', 'channel']]).map(outlier_lookup).to_numpy()
    plot_df = plot_df.loc[~is_outlier, :]

    with PdfPages(pdf_path) as pdf:
        palette = sns.color_palette('bright', n_colors=10)
        color_lookup = {
            'left, rostral array': palette[0],
            'right, rostral array': palette[1],
            'left, caudal array': palette[9],
            'right, caudal array': palette[8],
        }
        style_lookup = {
            'caudal': (),
            'rostral': (4, 4)
        }
        plot_df.loc[:, 'time_msec'] = plot_df['time_usec'] * 1e-3
        # this_iterator = tqdm(plot_df.groupby(['location', 'elecName']))
        this_iterator = tqdm(plot_df.groupby('location'))
        for name, group in this_iterator:
            '''if name not in ['ankle', 'mid_trunk', 'shoulder']:
                continue'''
            g = sns.relplot(
                data=group,
                x='time_msec', y='value',
                col='ycoords', row='xcoords',
                hue='side', style='elecName', palette='bright', dashes=style_lookup,
                # hue='Poke location / recording paddle', palette=color_lookup,
                kind='line', linewidth=1,
                facet_kws=dict(
                    margin_titles=True, sharey=True
                    ),
                height=1., aspect=1.5, **relplot_kwargs
                )
            for ax in g.axes.flatten():
                ax.axvline(0, color='r', linewidth=1.5)
                ax.axvline(10, color='k', linewidth=1., linestyle='--')
                # ax.set_ylim(-10, 10)
            g.set_titles(col_template='{col_name}')
            g.figure.suptitle(f"{name}", fontsize=10)
            # g.figure.tight_layout()
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if show_plots:
                plt.show()
            else:
                plt.close()
