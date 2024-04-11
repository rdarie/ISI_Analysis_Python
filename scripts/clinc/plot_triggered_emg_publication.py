import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
import os
import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from isicpy.lookup_tables import (
    eid_remix_lookup, eids_ordered_xy, eid_palette, dsi_channels)
from isicpy.clinc_lookup_tables import clinc_paper_matplotlib_rc, clinc_paper_emg_palette
from scipy import signal
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

sns.set(
    context='paper', style='white',
    palette='deep', font='sans-serif',
    font_scale=1, color_codes=True,
    rc=clinc_paper_matplotlib_rc
    )

from matplotlib import pyplot as plt

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
routing_config_info = pd.read_json(folder_path / 'analysis_metadata/routing_config_info.json')
file_name_list = routing_config_info['child_file_name'].to_list()

emg_dict = {}
envelope_dict = {}
stim_info_dict = {}
lfp_dict = {}
show_reref_lfp = True
zoom_in = False
zoom_suffix = '_zoom_in' if zoom_in else '_zoom_out'
reref_suffix = '_reref' if show_reref_lfp else ''

pdf_path = folder_path / "figures" / (f'emg_latency{reref_suffix}{zoom_suffix}.pdf')
png_path = folder_path / "figures" / (f'emg_latency{reref_suffix}{zoom_suffix}.png')
for file_name in file_name_list:
    # lfp, trial averaged
    stim_info_path = (file_name + '_stim_info.parquet')
    if not os.path.exists(folder_path / stim_info_path):
        continue
    print(f"Loading {file_name}...")
    emg_path = (file_name + '_epoched_emg.parquet')
    envelope_path = (file_name + '_epoched_envelope.parquet')
    if show_reref_lfp:
        lfp_path = (file_name + '_epoched_reref_lfp.parquet')
    else:
        lfp_path = (file_name + '_epoched_lfp.parquet')
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

this_eid_order = [cn for cn in eids_ordered_xy if cn in lfp_df.columns]
this_eid_palette = {lbl: eid_col for lbl, eid_col in eid_palette.items() if lbl in lfp_df.columns}

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

if False:
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
            g.set_ylabels('Normalized EMG (a.u.)')
            g._legend.set_title('Stim. amplitude (uA)')
            g.figure.align_labels()
            pdf.savefig()
            plt.close()

group_features = ['eid', 'freq', 'pw', 'amp']
zoom_in_lims = (-2, 30)
zoom_out_lims = (-25, 325)

emg_channel_order = [key for key in clinc_paper_emg_palette.keys()]
trial_info = lfp_df.index.to_frame().reset_index(drop=True)
with PdfPages(pdf_path) as pdf:
    for name, emg_group in emg_df.groupby(group_features):
        eid, freq, pw, amp = name
        if amp != 2100:
            continue
        if freq > 40:
            continue
        if eid != 43:
            continue
        print(f"On page {name}")
        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(2, 1)
        ax = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[1, 0]),
            ]
        emg_t = emg_group.index.get_level_values('t')
        if zoom_in:
            emg_mask = (emg_t >= zoom_in_lims[0] * 1e-3) & (emg_t <= zoom_in_lims[1] * 1e-3)
        else:
            emg_mask = (emg_t >= zoom_out_lims[0] * 1e-3) & (emg_t <= zoom_out_lims[1] * 1e-3)
        plot_emg = emg_group.loc[emg_mask, :].stack().reset_index().rename(columns={0: 'value'})
        plot_emg['t_msec'] = plot_emg['t'] * 1e3
        dy = 2.5
        y_offset = 0
        for chan in emg_channel_order:
            this_mask = plot_emg['channel'] == chan
            plot_emg.loc[this_mask, 'value'] += y_offset
            y_offset -= dy
        sns.lineplot(
            plot_emg, x='t_msec', y='value',
            hue='channel', palette=clinc_paper_emg_palette,
            hue_order=emg_channel_order,
            ax=ax[1],
            estimator='mean', errorbar='se', legend=zoom_in
            )
        ax[1].set_ylabel('Normalized EMG (a.u.)')
        ax[1].set_yticklabels([])
        ax[1].set_xlabel('Time (msec.)')
        if zoom_in:
            sns.move_legend(ax[1], 'center left', bbox_to_anchor=(1, 0.5))
            ax[1].legend_.set_title('EMG\nChannel')
            for legend_handle in ax[1].legend_.legendHandles:
                if isinstance(legend_handle, mpl.lines.Line2D):
                    legend_handle.set_lw(4 * legend_handle.get_lw())
        #
        lfp_group = lfp_df.xs(name, level=group_features)
        lfp_t = lfp_group.index.get_level_values('t')
        if zoom_in:
            artifact_mask = (lfp_t > 0) & (lfp_t <= 1e-3)
            lfp_group.loc[artifact_mask, :] = np.nan
        if zoom_in:
            lfp_mask = (
                (lfp_t >= zoom_in_lims[0] * 1e-3) &
                (lfp_t <= zoom_in_lims[1] * 1e-3)
            )
        else:
            lfp_mask = (
                (lfp_t >= zoom_out_lims[0] * 1e-3) &
                (lfp_t <= zoom_out_lims[1] * 1e-3) & (lfp_t.isin(lfp_t[::10]))
            )
        plot_lfp = lfp_group.loc[lfp_mask, :].stack().reset_index().rename(columns={0: 'value'})
        plot_lfp['t_msec'] = plot_lfp['t'] * 1e3
        plot_lfp['value'] = plot_lfp['value'] * 1e-3
        ## substitute renumbered EIDs
        this_eid_palette_remix = {eid_remix_lookup[old_name]: c for old_name, c in this_eid_palette.items()}
        this_eid_order_remix = [eid_remix_lookup[old_name] for old_name in this_eid_order]
        plot_lfp['channel'] = plot_lfp.apply(lambda x: eid_remix_lookup[x['channel']], axis='columns')
        #####
        dy = .25
        y_offset = 0
        for chan in this_eid_order_remix:
            this_mask = plot_lfp['channel'] == chan
            plot_lfp.loc[this_mask, 'value'] += y_offset
            y_offset -= dy
        sns.lineplot(
            plot_lfp, x='t_msec', y='value',
            ax=ax[0],
            estimator='mean', errorbar='se',
            hue='channel', palette=this_eid_palette_remix, hue_order=this_eid_order_remix,
            legend=zoom_in
            )
        if show_reref_lfp:
            ax[0].set_ylabel('Reref. Spinal Potential (mV)')
        else:
            ax[0].set_ylabel('Spinal Potential (mV)')
        ax[0].set_xlabel('')
        ax[0].set_xticklabels([])
        if zoom_in:
            ax[0].legend_.set_title('Spinal\nChannel')
            sns.move_legend(
                ax[0], 'center left', bbox_to_anchor=(1, 0.5),
                ncols=1 if show_reref_lfp else 2)
            for legend_handle in ax[0].legend_.legendHandles:
                if isinstance(legend_handle, mpl.lines.Line2D):
                    legend_handle.set_lw(4 * legend_handle.get_lw())
        for theax in ax:
            if zoom_in:
                # theax.axvspan(zoom_in_lims[0], zoom_in_lims[1], color='g', alpha=0.1)
                theax.axvspan(0, 1, color='r', zorder=2.005)
                theax.set_xlim(zoom_in_lims)
            else:
                theax.axvspan(zoom_in_lims[0], zoom_in_lims[1], color='g', alpha=0.1)
                theax.set_xlim(zoom_out_lims)
        if zoom_in:
            desired_figsize = (2.1, 3)
        else:
            desired_figsize = (2.4, 3)
        fig.set_size_inches(desired_figsize)
        fig.align_labels()
        # fig.suptitle(f'E{eid} {amp} uA {int(freq)} Hz {pw} usec PW')
        fig.tight_layout(pad=25e-2)
        pdf.savefig()
        fig.savefig(png_path)
        # plt.close()
        plt.show()

