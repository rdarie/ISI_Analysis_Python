import matplotlib as mpl
mpl.use('QTAgg')  # generate interactive output

from isicpy.third_party.pymatreader import hdf5todict
from pathlib import Path
import h5py
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set(
    context='talk', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True,
)

from matplotlib import pyplot as plt

folder_path = Path(r"/users/rdarie/20231109_temp/")
file_name = "MB_1699558933_985097_f.mat"
emg_block_name = "Block0001"
file_timestamp_parts = file_name.split('_')
file_start_time = pd.Timestamp(float('.'.join(file_timestamp_parts[1:3])), unit='s', tz='EST')

clinc_df = pd.read_parquet(folder_path / file_name.replace('.mat', '_clinc.parquet'))
clinc_df.index = clinc_df.index + file_start_time

this_routing = {
    'S10': 'E42',
    'S0_S2': 'E59',
    'S14': 'E45',
    'S12_S20': 'E36',
    'S11': 'E37',
    'S16': 'E44',
    'S18': 'E43',
    'S7': 'E48',
    'S1_S3': 'E41',
    'S23': 'E11',
    'S6': 'E3',
    'S22': 'E6',
    'S19': 'E18',
    'S15': 'E51'
}

clinc_df.rename(columns=this_routing, inplace=True)

clinc_trigs = pd.read_parquet(folder_path / file_name.replace('.mat', '_clinc_trigs.parquet'))
clinc_trigs.index = clinc_trigs.index + file_start_time
clinc_sample_rate = 36960

emg_df = pd.read_parquet(folder_path / f"{emg_block_name}_emg.parquet")
emg_sample_rate = 500
emg_df.columns = [
    'Right EDL', 'Right BF',
    'Right GAS', 'Left EDL',
    'Left BF', 'Left GAS'
    ]
dsi_trigs = pd.read_parquet(folder_path / f"{emg_block_name}_dsi_trigs.parquet")
dsi_trigs_sample_rate = 1000

dsi_coarse_offset = 115.44
dsi_fine_offset = 0
emg_df.index = emg_df.index + pd.Timedelta(dsi_coarse_offset, unit='s')

align_timestamps = [54474.0, 169646.5, 338779.0, 408505.0]

stim_parameters = [
    600, 900, 1200, 1500
]

t_zero = clinc_df.index[0]
clinc_df.index = (clinc_df.index - t_zero)
emg_df.index = (emg_df.index - t_zero)
window_len = 2500e-3  # milliseconds

clinc_window_n_samples = int(window_len * clinc_sample_rate)
t_clinc = np.arange(clinc_window_n_samples) / clinc_sample_rate
emg_window_n_samples = int(window_len * emg_sample_rate)
t_emg = np.arange(emg_window_n_samples) / emg_sample_rate

aligned_lfp = {}
aligned_emg = {}

left_sweep = 250  # msec
for ts_index, ts in enumerate(align_timestamps):
    this_lfp = pd.DataFrame(
        clinc_df.loc[clinc_df.index >= pd.Timedelta(ts - left_sweep, unit='ms'), :].iloc[:clinc_window_n_samples, :].to_numpy(),
        columns=clinc_df.columns, index=t_clinc
        )  ## converted to uV
    this_lfp.index.name = 't'
    aligned_lfp[stim_parameters[ts_index]] = this_lfp
    this_emg = pd.DataFrame(
        emg_df.loc[emg_df.index >= pd.Timedelta(ts - left_sweep, unit='ms'), :].iloc[:emg_window_n_samples, :].to_numpy() * 1e3,
        columns=emg_df.columns, index=t_emg
        ) ## converted to mV
    this_emg.index.name = 't'
    aligned_emg[stim_parameters[ts_index]] = this_emg

aligned_lfp_df = pd.concat(aligned_lfp, names=['amp'])
aligned_lfp_df.columns.name = 'channel'

aligned_emg_df = pd.concat(aligned_emg, names=['amp'])
aligned_emg_df.columns.name = 'channel'

plot_lfp_df = aligned_lfp_df.iloc[:, [0]].stack().to_frame(name='value').reset_index()
plot_lfp_df.loc[:, 't_msec'] = plot_lfp_df['t'] * 1e3

plot_emg_df = aligned_emg_df.stack().to_frame(name='value').reset_index()
plot_emg_df.loc[:, 't_msec'] = plot_emg_df['t'] * 1e3

add_dy = 100
current_y_offset = 0

for chan_name in aligned_emg_df.columns:
    plot_emg_df.loc[plot_emg_df['channel'] == chan_name, 'value'] += current_y_offset
    current_y_offset += add_dy

pdf_path = folder_path / file_name.replace('.mat', '_emg.pdf')
with PdfPages(pdf_path) as pdf:
    g = sns.relplot(
        data=plot_emg_df,
        col='amp',
        x='t', y='value', hue='channel', kind='line',
        lw=1, height=3, aspect=1)
    g.axes[0, 0].set_ylim(-250, 650)
    g.set_ylabels('EMG (mV)')
    g.set_xlabels('Time (sec)')
    g.set_titles(col_template='{col_name} uA')
    g.figure.set_size_inches(16, 4)
    g.figure.align_labels()
    pdf.savefig()
    plt.show()

pdf_path = folder_path / file_name.replace('.mat', '_one_ecap.pdf')
with PdfPages(pdf_path) as pdf:
    g = sns.relplot(
        data=plot_lfp_df, col='amp',
        x='t', y='value', hue='channel', kind='line',
        lw=1, height=3, aspect=2)
    # g.axes[0, 0].set_ylim(-250, 650)
    g.set_ylabels('Spinal LFP (mV)')
    g.set_xlabels('Time (sec)')
    g.set_titles(col_template='{col_name} uA')
    g.figure.set_size_inches(16, 2)
    g.figure.align_labels()
    pdf.savefig()
    plt.show()

