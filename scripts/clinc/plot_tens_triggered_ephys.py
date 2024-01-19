import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
import os
import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
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

apply_stim_blank = False
lfp_dict = {}
for file_name in file_name_list:
    # lfp, trial averaged
    '''
    lfp_path = (file_name + '_tens_epoched_lfp.parquet')
    pdf_path = folder_path / "figures" / ('tens_epoched_lfp.pdf')
    group_features = ['pw']
    relplot_kwargs = dict(
        estimator='mean', errorbar='se', hue='amp', palette='flare',
        col_order=[
            "E47", "E21", "E0", "E11", 'E58', "E53",
            "E16", "E6", "E59", "E45", "E37", "E42",
            ])
    '''
    # reref, trial averaged
    lfp_path = (file_name + '_tens_epoched_reref_lfp.parquet')
    pdf_path = folder_path / "figures" / ('tens_epoched_reref_lfp.pdf')
    group_features = ['pw']
    relplot_kwargs = dict(
        estimator='mean', errorbar='se', hue='amp', palette='crest',
        col_order=[
            "E47", "E0", 'E58', "E16", "E59", "E37",
            ]
        )
    lfp_dict[file_name] = pd.read_parquet(folder_path / lfp_path)

lfp_df = pd.concat(lfp_dict, names=['block'])
del lfp_dict

plot_df = lfp_df.stack().reset_index().rename(columns={0: 'value'})

recruitment_keys = ['pw', 'amp']
auc_df = lfp_df.abs().stack().groupby(recruitment_keys + ['block', 'timestamp']).mean()
g = sns.displot(data=auc_df.reset_index().rename(columns={0: 'auc'}), x='auc')
plt.show()

plot_df.loc[:, 'is_outlier'] = False
outlier_thresh = 20
for name, group in plot_df.groupby(recruitment_keys + ['block', 'timestamp']):
    plot_df.loc[group.index, 'is_outlier'] = auc_df.loc[name] > outlier_thresh

pw_lims = [0, 500e-6]
plot_df.loc[:, 't_msec'] = plot_df['t'] * 1e3
if apply_stim_blank:
    blank_mask = (plot_df['t'] > pw_lims[0]) & (plot_df['t'] < pw_lims[1])
    plot_df.loc[blank_mask, 'value'] = np.nan

t_min, t_max = plot_df['t_msec'].min(), plot_df['t_msec'].max()
plot_df = plot_df.loc[~plot_df['is_outlier'], :]

if not os.path.exists(folder_path / "figures"):
    os.makedirs(folder_path / "figures")

with PdfPages(pdf_path) as pdf:
    t_mask = (plot_df['t'] >= -25e-3) & (plot_df['t'] <= 100e-3)
    g = sns.relplot(
        data=plot_df.loc[t_mask, :],
        col='channel', row='location',
        x='t_msec', y='value',
        kind='line', height=4, aspect=1.8,
        facet_kws=dict(sharey=False, margin_titles=True),
        **relplot_kwargs
        )
    for ax in g.axes.flatten():
        ax.axvline(0, color='r')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    g.set_titles(row_template="TENS on\n{row_name} ankle")
    g.set_xlabels('Time (msec.)')
    g.set_ylabels('LFP (uV)')
    g._legend.set_title('TENS amplitude (V)')
    # g.figure.align_labels()
    pdf.savefig()
    plt.close()

'''
pdf_path = folder_path / "figures" / ('tens_epoched_lfp_per_amp.pdf')
group_features = ['pw']
relplot_kwargs = dict(
    estimator='mean', errorbar='se', hue='channel', palette='Spectral',
    hue_order=[
        "E47", "E21", "E0", "E11", 'E58', "E53",
        "E16", "E6", "E59", "E45", "E37", "E42",
        ],
    row='location',
    x='t_msec', y='value',
    kind='line', height=4, aspect=1.8,
    facet_kws=dict(sharey=True, margin_titles=True),
    )
dy = 12.5
'''
pdf_path = folder_path / "figures" / ('tens_epoched_reref_lfp_per_amp.pdf')
group_features = ['pw', 'amp']
relplot_kwargs = dict(
    estimator='mean', errorbar='se', hue='channel', palette='Paired',
    hue_order=[
        "E47", "E0", 'E58', "E16", "E59", "E37",
    ], row='location',
    x='t_msec', y='value',
    kind='line', height=4, aspect=1.8,
    facet_kws=dict(sharey=False, margin_titles=True),
)
dy = 7.5

y_offset = 0
for chan in relplot_kwargs['hue_order']:
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
        g.set_ylabels('LFP (uV)')
        g._legend.set_title('Channel')
        g.figure.suptitle(f'amp = {amp} V')
        # g.figure.align_labels()
        pdf.savefig()
        plt.close()
