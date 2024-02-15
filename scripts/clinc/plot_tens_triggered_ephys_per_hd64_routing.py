import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
import os
import pandas as pd
from pathlib import Path
from isicpy.lookup_tables import HD64_topo_list, HD64_topo, HD64_labels, eids_ordered_xy, eid_palette
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set(
    context='talk', style='darkgrid',
    palette='deep', font='sans-serif',
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

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202401251300-Phoenix")
routing_config_info = pd.read_json(folder_path / 'analysis_metadata/routing_config_info.json')
routing_config_info['config_start_time'] = routing_config_info['config_start_time'].apply(
    lambda x: pd.Timestamp(x, tz='GMT'))
routing_config_info['config_end_time'] = routing_config_info['config_end_time'].apply(
    lambda x: pd.Timestamp(x, tz='GMT'))


for yml_path, this_routing in routing_config_info.groupby('yml_path'):
    cfg_name = Path(yml_path).stem
    print(f'On config {cfg_name}')
    apply_stim_blank = False
    lfp_dict = {}
    lfp_type = "lfp"
    pdf_path = folder_path / "figures" / (f'tens_epoched_{lfp_type}_{cfg_name}.pdf')
    for file_name in this_routing['child_file_name']:
        # lfp, trial averaged
        lfp_path = (file_name + f'_tens_epoched_{lfp_type}.parquet')
        if not os.path.exists(folder_path / lfp_path):
            continue
        print(f'\tLoading {file_name}')
        group_features = ['pw']
        relplot_kwargs = dict(
            estimator='mean', errorbar='se', hue='amp',
        )
        relplot_kwargs['palette'] = 'crest' if ('reref' in lfp_type) else 'flare'
        ###
        lfp_dict[file_name] = pd.read_parquet(folder_path / lfp_path)
        this_eid_order = [cn for cn in eids_ordered_xy if cn in lfp_dict[file_name].columns]
        this_eid_palette = {lbl: eid_col for lbl, eid_col in eid_palette.items() if lbl in lfp_dict[file_name].columns}
        relplot_kwargs["col_order"] = this_eid_order
    if not len(lfp_dict):
        print('\tNo files for this cfg')
        continue
    lfp_df = pd.concat(lfp_dict, names=['block'])
    del lfp_dict

    plot_df = lfp_df.stack().reset_index().rename(columns={0: 'value'})

    recruitment_keys = ['pw', 'amp', 'location']
    auc_df = lfp_df.abs().stack().groupby(recruitment_keys + ['block', 'timestamp']).mean()

    #  g = sns.displot(data=auc_df.reset_index().rename(columns={0: 'auc'}), x='auc')
    #  g.axes[0][0].axvline(auc_df.mean() + 3 * auc_df.std(), color='r')
    #  g.figure.suptitle(f"{cfg_name}")
    #  plt.show()

    plot_df.loc[:, 'is_outlier'] = False
    outlier_thresh = auc_df.mean() + 3 * auc_df.std()
    print('\tIdentifying outliers...')
    for name, group in tqdm(plot_df.groupby(recruitment_keys + ['block', 'timestamp'])):
        plot_df.loc[group.index, 'is_outlier'] = auc_df.loc[name] > outlier_thresh
    print('\t\tDone.')
    pw_lims = [0, 500e-6]
    plot_df.loc[:, 't_msec'] = plot_df['t'] * 1e3
    if apply_stim_blank:
        blank_mask = (plot_df['t'] > pw_lims[0]) & (plot_df['t'] < pw_lims[1])
        plot_df.loc[blank_mask, 'value'] = np.nan

    t_min, t_max = plot_df['t_msec'].min(), plot_df['t_msec'].max()
    plot_t_min, plot_t_max = -1e-3, 50e-3
    plot_df = plot_df.loc[~plot_df['is_outlier'], :]

    if not os.path.exists(folder_path / "figures"):
        os.makedirs(folder_path / "figures")

    if True:
        with PdfPages(pdf_path) as pdf:
            t_mask = (plot_df['t'] >= plot_t_min) & (plot_df['t'] <= plot_t_max)
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
            if 'reref' in lfp_type:
                g.set_ylabels('Reref. LFP (uV)')
            else:
                g.set_ylabels('LFP (uV)')
            g._legend.set_title('TENS amplitude (V)')
            # g.figure.align_labels()
            pdf.savefig()
            plt.close()
            print(f"saved {pdf_path}")

    if True:
        pdf_path = folder_path / "figures" / (f'tens_epoched_{lfp_type}_per_amp_{cfg_name}.pdf')
        group_features = ['pw', 'amp']
        relplot_kwargs = dict(
            estimator='mean', errorbar='se', hue='channel', palette=eid_palette,  # palette='Paired',
            hue_order=this_eid_order,
            # hue_order=lfp_df.columns,
            row='location',
            x='t_msec', y='value',
            kind='line', height=4, aspect=1.8,
            facet_kws=dict(sharey=False, margin_titles=True),
        )

        dy = 10.
        y_offset = 0
        for chan in relplot_kwargs['hue_order']:
            this_mask = plot_df['channel'] == chan
            plot_df.loc[this_mask, 'value'] += y_offset
            y_offset -= dy

        with PdfPages(pdf_path) as pdf:
            t_mask = (plot_df['t'] >= plot_t_min) & (plot_df['t'] <= plot_t_max)
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
                if 'reref' in lfp_type:
                    g.set_ylabels('Reref. LFP (uV)')
                else:
                    g.set_ylabels('LFP (uV)')
                g._legend.set_title('Channel')
                g.figure.suptitle(f'TENS amplitude: {amp} V')
                # g.figure.align_labels()
                pdf.savefig()
                plt.close()
            print(f"saved {pdf_path}")
