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

folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
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

dsi_trigs = pd.read_parquet(folder_path / f"{emg_block_name}_dsi_trigs.parquet")
dsi_trigs_sample_rate = 1000

dsi_coarse_offset = 115.44
dsi_fine_offset = 0
t_zero = min(file_start_time, emg_df.index[0])
t_start_dsi = (emg_df.index[0] - t_zero).total_seconds() + dsi_coarse_offset + dsi_fine_offset

t_start_clinc = (file_start_time - t_zero).total_seconds()

align_timestamps = [54474.0, 169646.5, 338779.0, 408505.0]

stim_parameters = [
    600, 900, 1200, 1500
]

clinc_df.index = (clinc_df.index - clinc_df.index[0])

window_len = 40e-3  # milliseconds
window_n_samples = int(window_len * clinc_sample_rate)
t = np.arange(window_n_samples) / clinc_sample_rate

aligned_lfp = {}
for ts_index, ts in enumerate(align_timestamps):
    this_lfp = pd.DataFrame(
        clinc_df.loc[clinc_df.index >= pd.Timedelta(ts, unit='ms'), :].iloc[:window_n_samples, :].to_numpy(),
        columns=clinc_df.columns, index=t
        )  ## converted to uV
    this_lfp.index.name = 't'
    aligned_lfp[stim_parameters[ts_index]] = this_lfp

aligned_lfp_df = pd.concat(aligned_lfp, names=['amp'])
aligned_lfp_df.columns.name = 'channel'

plot_df = aligned_lfp_df.stack().to_frame(name='value').reset_index()

plot_df.loc[:, 't_msec'] = plot_df['t'] * 1e3

add_dy = .5
current_y_offset = 0

for chan_name in aligned_lfp_df.columns:
    plot_df.loc[plot_df['channel'] == chan_name, 'value'] += current_y_offset
    current_y_offset += add_dy

pdf_path = folder_path / file_name.replace('.mat', '_ecaps.pdf')
with PdfPages(pdf_path) as pdf:
    g = sns.relplot(
        data=plot_df,
        col='amp', col_wrap=2,
        x='t_msec', y='value', hue='channel', kind='line',
        lw=1)
    g.set_ylabels('Spinal LFP (mV)')
    g.set_xlabels('Time (msec)')
    g.set_titles(col_template='{col_name} uA')
    pdf.savefig()
    plt.show()

