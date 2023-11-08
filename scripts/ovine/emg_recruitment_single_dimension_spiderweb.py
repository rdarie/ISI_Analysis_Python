
import matplotlib as mpl
import os
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    mpl.use('Agg')   # generate postscript output
else:
    mpl.use('QT5Agg')   # generate interactive output
import traceback

# from isicpy.utils import load_synced_mat, makeFilterCoeffsSOS
# from isicpy.lookup_tables import emg_montages, muscle_names

from pathlib import Path
import pandas as pd
import numpy as np
import cloudpickle as pickle
import pdb
import gc
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_pdf import PdfPages
useDPI = 144
dpiFactor = 72 / useDPI
font_zoom_factor = 1.

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': .5,
        'lines.markersize': 1.5,
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
    context='paper', style='white',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True, rc=snsRCParams
)
for rcK, rcV in mplRCParams.items():
    mpl.rcParams[rcK] = rcV

import matplotlib as mpl
import os

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

if 'CCV_HEADLESS' in os.environ:
    mpl.use('Agg')   # generate postscript output
else:
    mpl.use('QT5Agg')   # generate interactive output

filterOptsNotch = {
    'line_noise': {
        'Wn': 60.,
        'nHarmonics': 1,
        'Q': 35,
        'N': 4,
        'btype': 'bandstop',
        'ftype': 'butter'
    },
}
filterOptsPost = {
    'low': {
        'Wn': 100.,
        'N': 4,
        'btype': 'low',
        'ftype': 'butter'
    },
}

folder_name = "202309141300-Phoenix"
blocks_list = [1, 2, 4]

# this_emg_montage = emg_montages['lower_v2']


data_path = Path(r"Z:\ISI\Phoenix") / f"{folder_name}"
pdf_folder = data_path / "figures_radu"
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)

blocks_list_str = '_'.join(f"{block_idx}" for block_idx in blocks_list)

x_axis_name = 'StimAmp'
if x_axis_name == 'freq':
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_emg_spiderweb_recruitment_freq.pdf")
    left_sweep = 0
    right_sweep = int(0.4 * 1e6)
    amp_cutoff = 9e3
    recruitment_keys = ['StimElec1', 'StimFreq', 'StimAmp']
    max_marker_color = 'm'
elif x_axis_name == 'freq_late':
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_emg_spiderweb_recruitment_freq_late.pdf")
    left_sweep = int(0.1 * 1e6)
    right_sweep = int(0.4 * 1e6)
    amp_cutoff = 10e3
    recruitment_keys = ['StimElec1', 'StimFreq', 'StimAmp']
    max_marker_color = 'm'
elif x_axis_name == 'StimAmp':
    pdf_path = pdf_folder / Path(f"Blocks_{blocks_list_str}_emg_spiderweb_recruitment_amp.pdf")
    left_sweep = 0
    right_sweep = int(0.1 * 1e6)
    freq_cutoff = 10
    recruitment_keys = ['StimElec1', 'StimFreq', 'StimAmp']
    max_marker_color = 'g'


all_emg = {}
all_kinematics = {}

for block_idx in blocks_list:
    kinematics_path = data_path / f"Block{block_idx:0>4d}_kinematics.parquet"
    this_kinematics_df = pd.read_parquet(kinematics_path)
    this_kinematics_df.set_index(
        ['TrialNum', 'stimElec', 'stimXCoord1', 'stimYCoord1', 'stimAmp', 'stimFreq', 'Channel'],
        inplace=True)
    emg_path = data_path / f"Block{block_idx:0>4d}_preproc_emg_epochs.parquet"
    this_emg_df = pd.read_parquet(emg_path)
    this_emg_df.set_index(
        ['TrialNum', 'StimElec1', 'StimXCoord1', 'StimYCoord1', 'StimAmp', 'StimFreq', 'ChannelID', 'Channel'],
        inplace=True)
    #
    this_emg_df.columns = pd.Index(np.arange(-1.95, 2, 150e-3), name='time')
    this_kinematics_df.columns = pd.Index(np.arange(-2, 2 + 50e-3, 50e-3), name='time')
    #
    all_emg[block_idx] = this_emg_df.stack().unstack(['Channel', 'ChannelID'])
    all_kinematics[block_idx] = this_kinematics_df.stack().unstack('Channel')

emg_df = pd.concat(all_emg, names=['block'])
from sklearn.preprocessing import minmax_scale
emg_df.loc[:, :] = minmax_scale(emg_df)

kinematics_df = pd.concat(all_kinematics, names=['block'])

emg_auc_df = emg_df.groupby(['block', 'TrialNum'] + recruitment_keys).mean()
average_auc_df = emg_auc_df.groupby(recruitment_keys).mean()
average_auc_df.columns = average_auc_df.columns.droplevel('ChannelID')

determine_side = lambda x: 'Left' if x[0] == 'L' else 'Right'

def polar_webmapper(
        data=None,
        azimuth='label_as_degree', radius='signal', hue='StimAmp',
        color_map=None, color_norm=None,
        marker_at_max=True, delta_deg=None, color=None):
    this_ax = plt.gca()
    plot_data = data.sort_values(azimuth)
    new_x_ticks = None
    for name, group in plot_data.groupby(hue):
        plot_vec = group.set_index(azimuth)[radius]
        az = (plot_vec.index + delta_deg / 2).to_list()
        r = plot_vec.to_list()
        if new_x_ticks is None:
            new_x_ticks = az.copy()
            new_x_labels = group['Channel'].to_list()
            new_x_labels = [lbl.replace('_', '\n') for lbl in new_x_labels]
        ## wrap around
        az.append(az[0])
        r.append(r[0])
        this_ax.plot(az, r, color=color_map(color_norm(name)))
    # pdb.set_trace()
    if marker_at_max:
        for this_deg, group in plot_data.groupby(azimuth):
            hue_of_max = group.set_index(hue)[radius].idxmax()
            this_ax.plot(
                this_deg + delta_deg / 2, group[radius].max(), '+',
                color=this_colormap(this_color_norm(hue_of_max)), zorder=6, )

    this_ax.set_xticks(new_x_ticks)
    this_ax.set_xticklabels(new_x_labels)
    this_ax.set_yticks([])
    # this_ax.set_yticklabels([])
    this_ax.tick_params(pad=snsRCParams['axes.labelpad'])
    return

show_plots = True
with PdfPages(pdf_path) as pdf:
    plot_auc = average_auc_df.stack().to_frame(name='signal').reset_index()
    ##
    plot_auc.loc[:, 'side'] = plot_auc['Channel'].apply(determine_side)
    ###
    elec_subset = plot_auc['StimElec1'].unique().tolist()
    label_subset = plot_auc['Channel'].unique().tolist()

    if x_axis_name == 'StimAmp':
        plot_auc.loc[:, 'amp_mA'] = plot_auc['StimAmp'] / 1e3

    elec_mask = plot_auc['StimElec1'].isin(elec_subset)
    label_mask = plot_auc['Channel'].isin(label_subset)
    delta_deg = 2 * np.pi / len(label_subset)
    labels_to_degrees = np.arange(delta_deg / 2, 2 * np.pi + delta_deg / 2, delta_deg)
    polar_map = {name: degree for name, degree in zip(label_subset, labels_to_degrees)}
    plot_auc.loc[:, 'label_as_degree'] = plot_auc['Channel'].map(polar_map)

    for this_freq, group in plot_auc.groupby('StimFreq'):
        plot_mask = elec_mask & label_mask & (plot_auc['StimFreq'] == this_freq)
        g = sns.FacetGrid(
            data=plot_auc.loc[plot_mask, :],
            col='StimElec1', col_wrap=6,
            # col='parent_elecConfig', row='elec_orientation',
            sharex=False, height=1., aspect=1,
            despine=False, subplot_kws=dict(projection='polar')
            )
        if x_axis_name in ['freq', 'freq_late']:
            this_colormap = sns.cubehelix_palette(
                start=0, rot=.4, dark=.2, light=.8,
                gamma=.75, as_cmap=True, reverse=True)
            x_axis_pretty_name = 'Stim.\nfreq. (Hz)'
            this_color_norm = plt.Normalize(
                vmin=plot_auc.loc[plot_mask, 'freq'].min() - 1e-6,
                vmax=plot_auc.loc[plot_mask, 'freq'].max() + 1e-6)
            g.map_dataframe(
                polar_webmapper, hue='freq',
                color_map=this_colormap, color_norm=this_color_norm,
                delta_deg=delta_deg)
            legend_hues = [7, 25, 50, 75, 100]
            legend_hues_normed = [this_color_norm(xx) for xx in legend_hues]
            custom_legend_text = [x_axis_pretty_name] + [
                f'{xx:.3g}' for xx in legend_hues
            ]
        elif x_axis_name == 'StimAmp':
            this_colormap = sns.cubehelix_palette(
                start=1.5, rot=.4, dark=.2, light=.8,
                gamma=.75, as_cmap=True, reverse=True)
            # this_color_norm = plt.Normalize(
            #     vmin=plot_auc.loc[plot_mask, 'amp_mA'].min() - 1e-6,
            #     vmax=plot_auc.loc[plot_mask, 'amp_mA'].max() + 1e-6)
            this_color_norm = plt.Normalize(vmin=plot_auc['amp_mA'].min(), vmax=plot_auc['amp_mA'].max())
            x_axis_pretty_name = 'Stim.\namp. (mA)'
            g.map_dataframe(
                polar_webmapper, hue='amp_mA',
                color_map=this_colormap, color_norm=this_color_norm,
                delta_deg=delta_deg)
            # num_legend_lines = 5
            # legend_hues_normed = np.linspace(0, 1, num_legend_lines)
            # legend_hues = [this_color_norm.inverse(xx) for xx in legend_hues_normed]
            legend_hues = [plot_auc['amp_mA'].min(), plot_auc['amp_mA'].max()]
            legend_hues_normed = [this_color_norm(xx) for xx in legend_hues]
            custom_legend_text = [x_axis_pretty_name] + [
                f'{xx:.2g}' for xx in legend_hues
            ]

        this_ax = plt.gca()

        dummy_legend_handle = mpl.patches.Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor='none', visible=False)
        custom_legend_lines = [dummy_legend_handle] + [
            Line2D([0], [0], color=this_colormap(xx), lw=1)
            for xx in legend_hues_normed
        ]
        this_ax.legend(
            custom_legend_lines, custom_legend_text,
            loc='lower left', bbox_to_anchor=(1.3, 1.025), borderaxespad=0.)

        g.set_titles(template="{col_name}")
        g.figure.suptitle(f'AUC vs {x_axis_name} ({this_freq} Hz)')
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if show_plots:
            plt.show()
        else:
            plt.close()
    
