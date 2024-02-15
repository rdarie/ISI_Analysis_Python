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
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set(
    context='talk', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True,
    )

from matplotlib import pyplot as plt
'''filterOpts = {
    'high': {
        'Wn': 1000.,
        'N': 2,
        'btype': 'high',
        'ftype': 'butter'
    },
}'''

folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/20231109-Phoenix")
# file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555"]
file_name_list = ["MB_1699560317_650555"]

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name_list = [
    'MB_1700670158_174163', 'MB_1700671071_947699', 'MB_1700671568_714180',
    'MB_1700672329_741498', 'MB_1700672668_26337', 'MB_1700673350_780580'
    ]

file_name_list = [
    'MB_1700672668_26337', 'MB_1700673350_780580']

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")
file_name_list = [
    "MB_1702047397_450767", "MB_1702048897_896568", "MB_1702049441_627410",
    "MB_1702049896_129326", "MB_1702050154_688487", "MB_1702051241_224335"
]
file_name_list = ["MB_1702050154_688487"]

# folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202401111300-Phoenix")
folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312201300-Phoenix")

routing_config_info = pd.read_json(folder_path / 'analysis_metadata/routing_config_info.json')
file_name_list = routing_config_info['child_file_name'].to_list()

apply_stim_blank = True
lfp_dict = {}
for file_name in file_name_list:
    # lfp, trial averaged
    lfp_path = (file_name + '_epoched_lfp.parquet')
    pdf_path = folder_path / "figures" / ('epoched_lfp.pdf')
    group_features = ['eid', 'pw']
    eid_order = [
        "E47", "E21", "E0", "E11",
        'E58', 'E53', "E16", "E6",
        "E59", "E45", "E37", "E42",
        ]
    relplot_kwargs = dict(
        estimator='mean', errorbar='se', hue='amp', palette='crest',
        col_order=None, col_wrap=4)

    # lfp, single trial
    '''
    lfp_path = (file_name + '_epoched_lfp.parquet')
    pdf_path = folder_path / "figures" / ('epoched_lfp_single.pdf')
    group_features = ['eid', 'pw', 'amp']
    relplot_kwargs = dict(estimator=None, units='timestamp')
    '''
    '''
    # reref, trial averaged
    lfp_path = (file_name + '_epoched_reref_lfp.parquet')
    pdf_path = folder_path / "figures" / ('epoched_reref_lfp.pdf')
    group_features = ['eid', 'pw']
    relplot_kwargs = dict(
        estimator='mean', errorbar='se', hue='amp',
        palette='crest', col_order=[
            "E47", "E0",
            'E58', "E16",
            "E59", "E37",
            ], col_wrap=3)
    '''
    # reref, single trial
    '''
    lfp_path = (file_name + '_epoched_reref_lfp.parquet')
    pdf_path = folder_path / "figures" / ('epoched_reref_lfp_single.pdf')
    group_features = ['eid', 'pw', 'amp']
    relplot_kwargs = dict(estimator=None, units='timestamp')
    '''

    # reref, trial averaged, per pulse
    '''
    lfp_path = (file_name + '_epoched_reref_lfp_per_pulse.parquet')
    pdf_path = folder_path / "figures" / ('epoched_reref_lfp_per_pulse.pdf')
    group_features = ['eid', 'pw']
    relplot_kwargs = dict(estimator='mean', errorbar='se', hue='amp')
    '''

    # lfp, trial averaged, per pulse
    '''
    lfp_path = (file_name + '_epoched_lfp_per_pulse.parquet')
    pdf_path = folder_path / "figures" / ('epoched_lfp_per_pulse.pdf')
    group_features = ['eid', 'pw']
    relplot_kwargs = dict(estimator='mean', errorbar='se', hue='amp')
    '''

    lfp_dict[file_name] = pd.read_parquet(folder_path / lfp_path)

lfp_df = pd.concat(lfp_dict, names=['block'])
del lfp_dict

plot_df = lfp_df.stack().reset_index().rename(columns={0: 'value'})

pw_lims = [0, 400e-6]
plot_df.loc[:, 't_msec'] = plot_df['t'] * 1e3
if apply_stim_blank:
    blank_mask = (plot_df['t'] > pw_lims[0]) & (plot_df['t'] < pw_lims[1])
    plot_df.loc[blank_mask, 'value'] = np.nan

t_min, t_max = plot_df['t_msec'].min(), plot_df['t_msec'].max()
dz0 = 1e-2  # 10 us min
if not os.path.exists(folder_path / "figures"):
    os.makedirs(folder_path / "figures")

pw_lims_lookup = {
    50: .200,
    150: .750
}
with PdfPages(pdf_path) as pdf:
    for name, group in plot_df.groupby(group_features):
        if 'amp' in group_features:
            eid, pw, amp = name
        else:
            eid, pw = name
        g = sns.relplot(
            data=group,
            col='channel',
            x='t_msec', y='value',
            kind='line', height=4, aspect=1,
            facet_kws=dict(sharey=True),
            **relplot_kwargs
            )
        for ax in g.axes.flatten():
            ax.set_xlim(-.5, 2)
            if apply_stim_blank:
                ax.axvspan(0, pw_lims_lookup[pw], color='r', zorder=2.005)
            else:
                ax.axvspan(0, pw_lims_lookup[pw], color='r', alpha=0.25)
        if 'amp' in group_features:
            g.figure.suptitle(f'amp: {amp} uA')
        g.set_titles(col_template="{col_name}")
        g.set_xlabels('Time (msec.)')
        g.set_ylabels('LFP (uV)')
        g._legend.set_title('Stim. amplitude (uA)')
        g.figure.suptitle("_".join([f"{nm}" for nm in name]))
        pdf.savefig()
        plt.close()
