import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from isicpy.clinc_lookup_tables import clinc_sample_rate, emg_sample_rate, dsi_trig_sample_rate
from scipy import signal
import numpy as np
from sklearn.preprocessing import minmax_scale
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set(
    context='notebook', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True,
    )

from matplotlib import pyplot as plt

'''filterOpts = {
    'high': {
        'Wn': 1000.,
        'N': 2,
        'btype': 'high',
        'ftype': 'butter'
    },
}'''


def reindex_and_interpolate(df, new_index):
    return df.reindex(df.index.union(new_index)).interpolate(method='index', limit_direction='both').loc[new_index]

def sanitize_triggers(
        srs, fill_val=0, thresh_opts=dict(), plotting=False):
    thresh_opts['plotting'] = plotting
    temp = srs.reset_index(drop=True)
    cross_index, cross_mask = getThresholdCrossings(temp, **thresh_opts)
    cross_mask.iloc[0] = True
    cross_mask.iloc[-1] = True
    cross_timestamps = pd.Series(srs.index[cross_mask])
    durations = pd.Series(cross_timestamps.diff().to_numpy(), index=cross_timestamps)
    valid_duration = pd.Series(False, index=cross_timestamps)
    fudge_factor = 0.05
    for target_duration in [pd.Timedelta(1, unit='s'), pd.Timedelta(2, unit='s')]:
        this_mask = (
            (durations > target_duration * (1 - fudge_factor)) &
            (durations < target_duration * (1 + fudge_factor))
            )
        valid_duration = valid_duration | this_mask
    valid_duration = reindex_and_interpolate(valid_duration, srs.index).fillna(method='ffill')
    if plotting:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(srs)
        ax[1].plot(valid_duration)
        plt.show()
    srs.loc[~valid_duration] = fill_val
    return srs


clinc_sample_interval_sec = float(clinc_sample_rate ** -1)
thresh_opts = dict(
    thresh=0.5, fs=1000, iti=None, absVal=False,
    keep_max=False, edgeType='both',
    plot_opts=dict(whichPeak=10, nSec=12))
plotting = False

folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/202311091300-Phoenix")
file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555"]

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name_list = [
    'MB_1700670158_174163', 'MB_1700671071_947699', 'MB_1700671568_714180',
    'MB_1700672668_26337', 'MB_1700673350_780580'
    ]

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")
file_name_list = [
    "MB_1702047397_450767", "MB_1702048897_896568", "MB_1702049441_627410",
    "MB_1702049896_129326", "MB_1702050154_688487", "MB_1702051241_224335"
]
file_name_list = ["MB_1702049441_627410", "MB_1702049896_129326"]

fine_offsets = {}
for file_name in file_name_list:
    print(file_name)
    fine_offsets[file_name] = {}
    clinc_trigs = pd.read_parquet(folder_path / (file_name + '_clinc_trigs.parquet'))['sync_wave']

    old_tmin, old_tmax = clinc_trigs.index[0], clinc_trigs.index[-1]
    downsampled_clinc_time = (
            old_tmin.round(freq='L') + pd.timedelta_range(
                0, old_tmax - old_tmin,
                freq=pd.Timedelta(1, unit='ms')
                )
            )
    clinc_orig = clinc_trigs.copy()
    clinc_trigs = reindex_and_interpolate(clinc_trigs, downsampled_clinc_time)

    '''fig, ax = plt.subplots()
    window_len = pd.Timedelta(12, unit='s')
    plot_mask = clinc_orig.index < clinc_orig.index[0] + window_len
    ax.plot(clinc_orig.loc[plot_mask])
    plot_mask = clinc_trigs.index < clinc_trigs.index[0] + window_len
    ax.plot(clinc_trigs)
    plt.show()'''

    with open(folder_path / 'dsi_block_lookup.json', 'r') as f:
        emg_block_list = json.load(f)[file_name]

    for emg_block_name in emg_block_list:
        dsi_trigs = pd.read_parquet(folder_path / f"{emg_block_name}_dsi_trigs.parquet").iloc[:, [1]]
        dsi_trigs.loc[:, :] = minmax_scale(dsi_trigs)
        dsi_trigs.loc[:, :] = (dsi_trigs > 0.5).astype(int)
        dsi_trigs = dsi_trigs.iloc[:, 0]

        tmin = max(dsi_trigs.index[0], clinc_trigs.index[0])
        tmax = min(dsi_trigs.index[-1], clinc_trigs.index[-1])

        masked_clinc = clinc_trigs.loc[(clinc_trigs.index >= tmin) & (clinc_trigs.index <= tmax)]
        masked_dsi = dsi_trigs.loc[(dsi_trigs.index >= tmin) & (dsi_trigs.index <= tmax)]

        lags = signal.correlation_lags(masked_clinc.shape[0], masked_dsi.shape[0], mode='full')
        xcorr = signal.correlate(masked_clinc.astype(int).to_numpy(), masked_dsi.astype(int).to_numpy(), mode='full')

        mask = (lags > -1501) & (lags < 1501)
        xcorr_srs = pd.Series(xcorr[mask], index=lags[mask])
        optimal_lag_samples = xcorr_srs.idxmax()
        print(f"the optimal lag is {optimal_lag_samples} samples")

        optimal_lag = optimal_lag_samples * 1e-3
        fine_offsets[file_name][emg_block_name] = optimal_lag

        if plotting:
            fig, ax = plt.subplots()
            ax.plot(xcorr_srs)
            ax.plot(optimal_lag_samples, xcorr_srs.loc[optimal_lag_samples], 'r*')
            plt.show()
            fig, ax = plt.subplots()
            ax.plot(clinc_trigs)
            plot_dsi = dsi_trigs.copy()
            plot_dsi.index = plot_dsi.index + pd.Timedelta(optimal_lag, unit='s')
            ax.plot(plot_dsi)
            plt.show()
        print('\tDone')

    with open(folder_path / 'dsi_to_mb_fine_offsets.json', 'w') as f:
        json.dump(fine_offsets, f, indent=4)
