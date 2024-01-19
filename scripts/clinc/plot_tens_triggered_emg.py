import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
import os
import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from isicpy.lookup_tables import dsi_channels
from isicpy.clinc_lookup_tables import clinc_sample_rate, emg_sample_rate, dsi_trig_sample_rate
from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set(
    context='talk', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True,
    rc={"xtick.bottom": True}
    )

'''
filterOpts = {
    'high': {
        'Wn': 1000.,
        'N': 2,
        'btype': 'high',
        'ftype': 'butter'
    },
}
'''

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")
file_name_list = ["MB_1702049441_627410", "MB_1702049896_129326"]

emg_dict = {}
envelope_dict = {}
tens_info_dict = {}
for file_name in file_name_list:
    emg_path = (file_name + '_tens_epoched_emg.parquet')
    envelope_path = (file_name + '_tens_epoched_envelope.parquet')
    tens_info_path = (file_name + '_tens_info.parquet')
    emg_dict[file_name] = pd.read_parquet(folder_path / emg_path)
    envelope_dict[file_name] = pd.read_parquet(folder_path / envelope_path)
    tens_info_dict[file_name] = pd.read_parquet(folder_path / tens_info_path)

emg_df = pd.concat(emg_dict, names=['block'])
del emg_dict
envelope_df = pd.concat(envelope_dict, names=['block'])
del envelope_dict
tens_info_df = pd.concat(tens_info_dict, names=['block'])
del tens_info_dict

emg_df.rename(columns=dsi_channels, inplace=True)
envelope_df.rename(columns=dsi_channels, inplace=True)

plot_df = emg_df.stack().reset_index().rename(columns={0: 'value'})
plot_df.loc[:, 't_msec'] = plot_df['t'] * 1e3
plot_df['value'] *= 1e3  # ???

t_min, t_max = plot_df['t_msec'].min(), plot_df['t_msec'].max()

if not os.path.exists(folder_path / "figures"):
    os.makedirs(folder_path / "figures")

pdf_path = folder_path / "figures" / ('tens_epoched_emg.pdf')
relplot_kwargs = dict(
    estimator='mean', errorbar='se', hue='amp',
    palette='viridis')

with PdfPages(pdf_path) as pdf:
    t_mask = (plot_df['t'] >= -25e-3) & (plot_df['t'] <= 100e-3)
    g = sns.relplot(
        data=plot_df.loc[t_mask, :],
        col='channel', row='location',
        x='t_msec', y='value',
        kind='line', height=4, aspect=1.8,
        facet_kws=dict(sharey=False, margin_titles=True),
        **relplot_kwargs)
    for ax in g.axes.flatten():
        ax.axvline(0, color='r')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    g.set_titles(row_template="TENS on\n{row_name} ankle")
    g.set_xlabels('Time (msec.)')
    g.set_ylabels('EMG (uV)')
    g._legend.set_title('TENS amplitude (V)')
    g.figure.align_labels()
    pdf.savefig()
    plt.close()

pdf_path = folder_path / "figures" / ('tens_epoched_emg_per_amp.pdf')
group_features = ['pw', 'amp']
colors_to_use = sns.color_palette('pastel', n_colors=3) + sns.color_palette('dark', n_colors=3)

relplot_kwargs = dict(
    estimator='mean', errorbar='se', hue='channel',
    palette={key: colors_to_use[idx] for idx, key in enumerate(emg_df.columns)},
    row='location', #  hue_order=emg_df.columns.to_list(),
    x='t_msec', y='value',
    kind='line', height=4, aspect=1.8,
    facet_kws=dict(sharey=False, margin_titles=True),
)

dy = 10
y_offset = 0
for chan in emg_df.columns:
    this_mask = plot_df['channel'] == chan
    plot_df.loc[this_mask, 'value'] += y_offset
    y_offset -= dy

with PdfPages(pdf_path) as pdf:
    t_mask = (plot_df['t'] >= -25e-3) & (plot_df['t'] <= 100e-3)
    for amp, group in plot_df.loc[t_mask, :].groupby('amp'):
        g = sns.relplot(
            data=group,
            **relplot_kwargs
            )
        for ax in g.axes.flatten():
            ax.axvline(0, color='r')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        g.set_titles(row_template="TENS on\n{row_name} ankle")
        g.set_xlabels('Time (msec.)')
        g.set_ylabels('EMG (uV)')
        g._legend.set_title('Channel')
        g.figure.suptitle(f'amp = {amp} V')
        # g.figure.align_labels()
        pdf.savefig()
        plt.close()
