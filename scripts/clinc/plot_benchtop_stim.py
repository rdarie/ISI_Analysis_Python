import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
import pandas as pd
import numpy as np
from pathlib import Path

import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import os
import re
from glob import glob

sns.set(
    context='paper', style='white',
    palette='deep', font='sans-serif',
    font_scale=1, color_codes=True,
    rc={
        'figure.dpi': 300, 'savefig.dpi': 300,
        'lines.linewidth': .5,
        'lines.markersize': 2.,
        'patch.linewidth': .5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        "xtick.bottom": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 1,
        "axes.labelsize": 5,
        "axes.titlesize": 6,
        "axes.titlepad": 1,
        "xtick.labelsize": 5,
        'xtick.major.pad': 1,
        'xtick.major.size': 2,
        "ytick.labelsize": 5,
        "ytick.major.pad": 1,
        'ytick.major.size': 2,
        "legend.fontsize": 5,
        "legend.title_fontsize": 6,
        }
    )

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202401261300-CLINC")

amp_sweep_data_dict = {}
for file_path in glob(f'{folder_path / "amp_sweep*.csv"}'):
    info = re.match('amp_sweep_(\d+)steps_(\d+)usec', Path(file_path).stem)
    amp = 12 * int(info.groups()[0])
    pw = 10 * int(info.groups()[1])
    this_data = pd.read_csv(file_path, header=9, names=['t', 'ch2']).set_index('t')
    amp_sweep_data_dict[(amp, pw)] = this_data
amp_sweep_data = pd.concat(amp_sweep_data_dict, names=['amp', 'pw'])
del amp_sweep_data_dict

pw_sweep_data_dict = {}
for file_path in glob(f'{folder_path / "pw_sweep*.csv"}'):
    info = re.match('pw_sweep_(\d+)steps_(\d+)ua', Path(file_path).stem)
    pw = 10 * int(info.groups()[0])
    amp = 12 * int(info.groups()[1])
    this_data = pd.read_csv(file_path, header=9, names=['t', 'ch2']).set_index('t')
    pw_sweep_data_dict[(amp, pw)] = this_data
pw_sweep_data = pd.concat(pw_sweep_data_dict, names=['amp', 'pw'])
del pw_sweep_data_dict

multi_freq = (
    pd.read_csv(
        folder_path / "multi_freq_120uA_200usec.csv",
        header=9, names=['t', 'ch1', 'ch2'])
    .set_index('t')
    )

if not os.path.exists(folder_path / 'figures'):
    os.makedirs(folder_path / 'figures')

pdf_path = folder_path / 'figures' / 'amp_sweep.pdf'
with PdfPages(pdf_path) as pdf:
    plot_df = amp_sweep_data.reset_index()
    plot_df['t_msec'] = plot_df['t'] * 1e3
    dx, dy = 0, 0
    x_offset, y_offset = 0, 0
    for name, group in plot_df.groupby('amp'):
        plot_df.loc[group.index, 't_msec'] = group['t_msec'] + x_offset
        plot_df.loc[group.index, 'ch2'] = group['ch2'] + y_offset
        x_offset += dx
        y_offset += dy
    t_min, t_max = plot_df['t_msec'].min(), plot_df['t_msec'].max()
    g = sns.relplot(
        x='t_msec', y='ch2',
        hue='amp', palette='flare',
        data=plot_df,
        kind='line', errorbar='se',
        facet_kws=dict(legend_out=False),
        )
    desired_figsize = (1.8, 1.4)
    g.figure.set_size_inches(desired_figsize)
    g.axes[0][0].set_xlim(t_min, t_max)
    g.set_ylabels('Voltage (V)')
    g.set_xlabels('Time (msec.)')
    g.legend.set_title('Amplitude (uA)')
    g.legend._set_loc(7)
    g.legend.set_bbox_to_anchor(
        (1., 0.5), transform=g.figure.transFigure)
    g.figure.draw_without_rendering()
    legend_approx_width = g.legend.legendPatch.get_width() / g.figure.get_dpi()  # inches
    new_right_margin = 1 - legend_approx_width / desired_figsize[0]
    g.figure.tight_layout(pad=5e-1, rect=[0, 0, new_right_margin, 1])
    # g.figure.subplots_adjust(
    #     right=1 - legend_approx_width / desired_figsize[0])
    g.figure.align_labels()
    pdf.savefig(pad_inches=25e-3)
    g.figure.savefig(folder_path / 'figures' / 'amp_sweep.png', pad_inches=25e-3)
    plt.close()

pdf_path = folder_path / 'figures' / 'pw_sweep.pdf'
with PdfPages(pdf_path) as pdf:
    plot_df = pw_sweep_data.reset_index()
    plot_df['t_msec'] = plot_df['t'] * 1e3
    mask = plot_df['pw'] < 1500
    plot_df = plot_df.loc[mask, :]
    dx, dy = 1e-1, -1e-1
    x_offset, y_offset = 0, 0
    for name, group in plot_df.groupby('pw'):
        plot_df.loc[group.index, 't_msec'] = group['t_msec'] + x_offset
        plot_df.loc[group.index, 'ch2'] = group['ch2'] + y_offset
        x_offset += dx
        y_offset += dy
    t_min, t_max = plot_df['t_msec'].min(), plot_df['t_msec'].max()
    plot_t_min, plot_t_max = t_min, 8
    g = sns.relplot(
        x='t_msec', y='ch2',
        hue='pw', palette='crest',
        data=plot_df,
        kind='line', errorbar='se',
        facet_kws=dict(legend_out=False),
    )
    desired_figsize = (1.8, 1.4)
    g.figure.set_size_inches(desired_figsize)
    g.axes[0][0].set_xlim(plot_t_min, plot_t_max)
    g.set_ylabels('Voltage (V)')
    g.set_xlabels('Time (msec.)')
    g.legend.set_title('Pulse\nWidth (usec.)')
    g.legend._set_loc(7)
    g.legend.set_bbox_to_anchor(
        (1., 0.5), transform=g.figure.transFigure)
    g.figure.draw_without_rendering()
    legend_approx_width = g.legend.legendPatch.get_width() / g.figure.get_dpi()  # inches
    new_right_margin = 1 - legend_approx_width / desired_figsize[0]
    g.figure.tight_layout(pad=5e-1, rect=[0, 0, new_right_margin, 1])
    # g.figure.subplots_adjust(
    #     right=1 - legend_approx_width / desired_figsize[0])
    g.figure.align_labels()
    pdf.savefig(pad_inches=25e-3)
    g.figure.savefig(folder_path / 'figures' / 'pw_sweep.png', pad_inches=25e-3)
    plt.close()

pdf_path = folder_path / 'figures' / 'multi_freq.pdf'
with PdfPages(pdf_path) as pdf:
    plot_df = multi_freq.stack().reset_index()
    plot_df.columns = ['t', 'channel', 'voltage']
    plot_df['t_msec'] = plot_df['t'] * 1e3
    dx, dy = 0, 15e-2
    x_offset, y_offset = 0, 0
    for name, group in plot_df.groupby('channel'):
        plot_df.loc[group.index, 't_msec'] = group['t_msec'] + x_offset
        plot_df.loc[group.index, 'voltage'] = group['voltage'] + y_offset
        x_offset += dx
        y_offset += dy
    t_min, t_max = plot_df['t_msec'].min(), plot_df['t_msec'].max()
    plot_t_min, plot_t_max = t_min, t_max
    g = sns.relplot(
        x='t_msec', y='voltage',
        hue='channel', palette='crest',
        data=plot_df,
        kind='line', errorbar='se',
        facet_kws=dict(legend_out=False),
    )
    desired_figsize = (3.6, 0.7)
    g.figure.set_size_inches(desired_figsize)
    g.axes[0][0].set_xlim(t_min, t_max)
    g.set_ylabels('Voltage (V)')
    g.set_xlabels('Time (msec.)')
    g.legend.set_title('Channel')
    g.legend._set_loc(7)
    g.legend.set_bbox_to_anchor(
        (1., 0.5), transform=g.figure.transFigure)
    g.figure.draw_without_rendering()
    legend_approx_width = g.legend.legendPatch.get_width() / g.figure.get_dpi()  # inches
    new_right_margin = 1 - legend_approx_width / desired_figsize[0]
    g.figure.tight_layout(pad=5e-1, rect=[0, 0, new_right_margin, 1])
    # g.figure.subplots_adjust(
    #     right=1 - legend_approx_width / desired_figsize[0])
    g.figure.align_labels()
    pdf.savefig(pad_inches=25e-3)
    g.figure.savefig(folder_path / 'figures' / 'multi_freq.png', pad_inches=25e-3)
    plt.close()
