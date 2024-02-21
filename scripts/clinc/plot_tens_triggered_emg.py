import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
import os
import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from isicpy.lookup_tables import dsi_channels
from isicpy.clinc_lookup_tables import clinc_paper_matplotlib_rc, clinc_paper_emg_palette
from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

sns.set(
    context='paper', style='white',
    palette='deep', font='sans-serif',
    font_scale=1, color_codes=True,
    rc=clinc_paper_matplotlib_rc
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
routing_config_info = pd.read_json(folder_path / 'analysis_metadata/routing_config_info.json')
routing_config_info['config_start_time'] = routing_config_info['config_start_time'].apply(
    lambda x: pd.Timestamp(x, tz='GMT'))
routing_config_info['config_end_time'] = routing_config_info['config_end_time'].apply(
    lambda x: pd.Timestamp(x, tz='GMT'))

emg_dict = {}
envelope_dict = {}
tens_info_dict = {}
for file_name in routing_config_info['child_file_name']:
    emg_path = (file_name + '_tens_epoched_emg.parquet')
    if not os.path.exists(folder_path / emg_path):
        continue
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

# emg_df.rename(columns=dsi_channels, inplace=True)
# envelope_df.rename(columns=dsi_channels, inplace=True)

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
    g.set_titles(row_template="TENS on\n{row_name} fetlock")
    g.set_xlabels('Time (msec.)')
    g.set_ylabels('EMG (uV)')
    g.legend.set_title('TENS amplitude (V)')
    g.figure.align_labels()
    pdf.savefig()
    plt.close()

pdf_path = folder_path / "figures" / ('tens_epoched_emg_per_amp.pdf')
group_features = ['pw', 'amp']

plot_t_min, plot_t_max = -10e-3, 80e-3
relplot_kwargs = dict(
    estimator='mean', errorbar='se', hue='channel',
    palette=clinc_paper_emg_palette,
    col='location', hue_order=[key for key in clinc_paper_emg_palette.keys()],
    x='t_msec', y='value',
    kind='line',  # height=4, aspect=1.8,
    facet_kws=dict(
        sharey=False, margin_titles=True, legend_out=False,
        xlim=(plot_t_min * 1e3, plot_t_max * 1e3),),
)

dy = 10
y_offset = 0
for chan in emg_df.columns:
    this_mask = plot_df['channel'] == chan
    plot_df.loc[this_mask, 'value'] += y_offset
    y_offset -= dy

with PdfPages(pdf_path) as pdf:
    t_mask = (plot_df['t'] >= plot_t_min) & (plot_df['t'] <= plot_t_max)
    for amp, group in plot_df.loc[t_mask, :].groupby('amp'):
        if amp != 25:
            continue
        g = sns.relplot(
            data=group,
            **relplot_kwargs
            )
        for ax in g.axes.flatten():
            ax.axvline(0, color='r')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        # g.set_titles(col_template="TENS on\n{col_name} fetlock")
        g.set_titles(col_template="")
        g.set_xlabels('Time (msec.)')
        g.set_ylabels('EMG (uV)')
        g.legend.set_title('EMG\nChannel')
        g.figure.suptitle(f'TENS amplitude: {amp} V', fontsize=1)
        desired_figsize = (4.8, 1.8)
        g.figure.set_size_inches(desired_figsize)
        sns.move_legend(
            g, 'center right', bbox_to_anchor=(1, 0.5),
            ncols=1)
        for legend_handle in g.legend.legendHandles:
            if isinstance(legend_handle, mpl.lines.Line2D):
                legend_handle.set_lw(4 * legend_handle.get_lw())

        g.figure.draw_without_rendering()
        legend_approx_width = g.legend.legendPatch.get_width() / g.figure.get_dpi()  # inches
        # new_right_margin = 1 - legend_approx_width / desired_figsize[0]
        new_right_margin = .825  # hardcode to align to lfp figure
        g.figure.subplots_adjust(right=new_right_margin)
        g.tight_layout(pad=25e-2, rect=[0, 0, new_right_margin, 1])
        g.figure.align_labels()
        pdf.savefig()
        plt.close()
