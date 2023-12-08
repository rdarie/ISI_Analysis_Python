import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
import os
import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings, mapToDF
from isicpy.lookup_tables import HD64_topo_list
from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler
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

clinc_sample_interval = pd.Timedelta(27077, unit='ns').to_timedelta64()
clinc_sample_interval_sec = float(clinc_sample_interval) * 1e-9
#  = clinc_sample_interval_sec ** -1
clinc_sample_rate = 36931.8
filterOpts = {
    'low': {
        'Wn': 100.,
        'N': 2,
        'btype': 'low',
        'ftype': 'butter'
    },
}
filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), clinc_sample_rate)

downsample_factor = 10
downsampled_interval = clinc_sample_interval_sec * downsample_factor
folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311071300-Phoenix")
file_name_list = [
    "MB_1699382682_316178", "MB_1699383052_618936", "MB_1699383757_778055", "MB_1699384177_953948",
    "MB_1699382925_691816", "MB_1699383217_58381", "MB_1699383957_177840"
    ]
dsi_block_list = []
file_name = "MB_1699384177_953948"

lfp_path = (file_name + '_clinc.parquet')
pdf_path = folder_path / "figures" / ('treadmill_illustration.pdf')
lfp_df = pd.read_parquet(folder_path / lfp_path)
lfp_df.index = lfp_df.index.total_seconds()
lfp_df = pd.DataFrame(
    signal.sosfiltfilt(filterCoeffs, lfp_df, axis=0),
    index=lfp_df.index, columns=lfp_df.columns).iloc[::downsample_factor, :]
implant_map = mapToDF("/users/rdarie/isi_analysis/ISI_Analysis_Python/ripple_map_files/hd64_square.map")
HD64_topo = pd.DataFrame(HD64_topo_list)

eids_present = HD64_topo.applymap(lambda x: f"E{x:0>2d}" in lfp_df.columns)

# fig, ax = plt.subplots()
# sns.heatmap(eids_present, ax=ax)

window_len = 10  # sec
window_len_samples = int(window_len / downsampled_interval)

align_timestamps = {
    1: 'off',
    33: 'on'
}

epoched_dict = {}
t = None
for ts, treadmill in align_timestamps.items():
    this_mask = lfp_df.index > ts
    first_index = np.flatnonzero(this_mask)[0]
    epoched_dict[(ts, treadmill)] = lfp_df.iloc[first_index:first_index + window_len_samples, :].copy()
    if t is None:
        t = np.arange(epoched_dict[(ts, treadmill)].shape[0]) * downsampled_interval
    epoched_dict[(ts, treadmill)].index = t
    epoched_dict[(ts, treadmill)].index.name = 't'

epoched_df = pd.concat(epoched_dict, names=['timestamp', 'treadmill'])
epoched_df = epoched_df - epoched_df.mean()

average_std = epoched_df.std().mean()
dy = 0
for cn in epoched_df.columns:
    epoched_df[cn] = epoched_df[cn] + dy
    dy += 10 * average_std

plot_df = epoched_df.stack().reset_index().rename(columns={0: 'value'})

if not os.path.exists(folder_path / "figures"):
    os.makedirs(folder_path / "figures")
with PdfPages(pdf_path) as pdf:
    g = sns.relplot(
        data=plot_df,
        col='treadmill',
        x='t', y='value',
        hue='eid',
        kind='line',
        # facet_kws=dict(sharey=False),
        )
    # plt.show()
    pdf.savefig()
    plt.close()
