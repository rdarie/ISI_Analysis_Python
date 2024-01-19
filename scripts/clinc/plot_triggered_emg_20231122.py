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
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set(
    context='talk', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True,
    )

from matplotlib import pyplot as plt

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name_list = [
    'MB_1700670158_174163', 'MB_1700671071_947699', 'MB_1700671568_714180',
    'MB_1700672668_26337', 'MB_1700673350_780580'
    ]

file_name_list = [
    'MB_1700672668_26337', 'MB_1700673350_780580']

emg_dict = {}
envelope_dict = {}
stim_info_dict = {}
lfp_dict = {}
for file_name in file_name_list:
    # lfp, trial averaged
    emg_path = (file_name + '_epoched_emg.parquet')
    envelope_path = (file_name + '_epoched_envelope.parquet')
    stim_info_path = (file_name + '_stim_info_per_pulse.parquet')
    lfp_path = (file_name + '_epoched_reref_lfp.parquet')
    emg_dict[file_name] = pd.read_parquet(folder_path / emg_path)
    envelope_dict[file_name] = pd.read_parquet(folder_path / envelope_path)
    stim_info_dict[file_name] = pd.read_parquet(folder_path / stim_info_path)
    lfp_dict[file_name] = pd.read_parquet(folder_path / lfp_path)

emg_df = pd.concat(emg_dict, names=['block'])
del emg_dict
envelope_df = pd.concat(envelope_dict, names=['block'])
del envelope_dict
stim_info_df = pd.concat(stim_info_dict, names=['block'])
del stim_info_dict
lfp_df = pd.concat(lfp_dict, names=['block'])
del lfp_dict

emg_df.rename(columns=dsi_channels, inplace=True)
envelope_df.rename(columns=dsi_channels, inplace=True)

plot_df = emg_df.stack().reset_index().rename(columns={0: 'value'})
plot_df.loc[:, 't_msec'] = plot_df['t'] * 1e3

plot_df['side'] = plot_df['channel'].apply(lambda x: x.split(' ')[0])
plot_df['muscle'] = plot_df['channel'].apply(lambda x: x.split(' ')[1])

t_min, t_max = plot_df['t_msec'].min(), plot_df['t_msec'].max()
dz0 = 1e-2  # 10 us min
if not os.path.exists(folder_path / "figures"):
    os.makedirs(folder_path / "figures")

relplot_kwargs = dict(estimator='mean', errorbar=None, hue='amp', palette='flare')
group_features = ['eid', 'freq', 'pw']

color_norm = Normalize(vmin=plot_df['amp'].min(), vmax=plot_df['amp'].max())
hue_mapper = cm.ScalarMappable(norm=color_norm, cmap='flare')

pdf_path = folder_path / "figures" / ('epoched_emg.pdf')
with PdfPages(pdf_path) as pdf:
    for name, group in plot_df.groupby(group_features):
        g = sns.relplot(
            data=group,
            col='side', col_order=['Left', 'Right'],
            row='muscle', row_order=['BF', 'GAS', 'EDL'],
            x='t_msec', y='value',
            kind='line',
            facet_kws=dict(sharey=True, margin_titles=True),
            alpha=0.2, height=3.5, aspect=1,
            **relplot_kwargs
            )
        for (muscle, side), ax in g.axes_dict.items():
            chan = f"{side} {muscle}"
            env_group = envelope_df.xs(name, level=group_features)[chan]
            for amp, subgroup in env_group.groupby('amp'):
                y = subgroup.groupby('t').mean()
                x = y.index * 1e3
                ax.plot(x, y, color=hue_mapper.to_rgba(amp))
                ax.set_yticklabels([])
                ax.axvspan(0, 1, color='r', zorder=2.005)
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.set_xlabels('Time (msec.)')
        g.set_ylabels('EMG (a.u.)')
        g._legend.set_title('Stim. amplitude (uA)')
        g.figure.align_labels()
        pdf.savefig()
        plt.show()

group_features = ['eid', 'freq', 'pw', 'amp']
pdf_path = folder_path / "figures" / ('emg_latency.pdf')
with PdfPages(pdf_path) as pdf:
    for name, emg_group in emg_df.groupby(group_features):
        eid, freq, pw, amp = name
        if amp < 1600:
            continue
        if freq < 40 or freq > 60:
            continue
        fig = plt.figure(constrained_layout=True)
        fig.set_size_inches(6, 12)
        gs = gridspec.GridSpec(
            2, 1, left=0.2, right=0.8)
        ax = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[1, 0]),
            ]
        emg_t = emg_group.index.get_level_values('t')
        emg_mask = (emg_t >= -2e-3) & (emg_t <= 20e-3)
        plot_emg = emg_group.loc[emg_mask, :].stack().reset_index().rename(columns={0: 'value'})
        plot_emg['t_msec'] = plot_emg['t'] * 1e3
        sns.lineplot(
            plot_emg, x='t_msec', y='value',
            hue='channel', palette='pastel', ax=ax[1],
            estimator='mean', errorbar='se'
            )
        ax[1].set_ylabel('EMG (a.u.)')
        ax[1].set_yticklabels([])
        ax[1].set_xlabel('Time (msec.)')
        #
        lfp_group = lfp_df.xs(name, level=group_features)
        lfp_t = lfp_group.index.get_level_values('t')
        lfp_mask = (lfp_t > 0) & (lfp_t <= 1e-3)
        lfp_group.loc[lfp_mask, :] = np.nan
        plot_lfp = lfp_group.stack().reset_index().rename(columns={0: 'value'})
        plot_lfp['t_msec'] = plot_lfp['t'] * 1e3
        sns.lineplot(
            plot_lfp, x='t_msec', y='value',
            hue='channel', palette='dark', ax=ax[0],
            estimator='mean', errorbar='se',
            hue_order=["E47", "E0", 'E58', "E16", "E59", "E37"]
            )
        ax[0].set_ylabel('LFP (uV)')
        ax[0].set_xlabel('')
        ax[0].set_xticklabels([])
        for theax in ax:
            theax.axvspan(0, 1, color='r', zorder=2.005)
            theax.set_xlim(-2, 20)
        fig.align_labels()
        pdf.savefig()
        plt.show()
        break
