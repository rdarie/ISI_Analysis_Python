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
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set(
    context='talk', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True,
    )


# folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202401111300-Phoenix")
folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312211300-Phoenix")

routing_config_info = pd.read_json(folder_path / 'analysis_metadata/routing_config_info.json')
file_name_list = routing_config_info['child_file_name'].to_list()

apply_stim_blank = True
emg_dict = {}
for file_name in file_name_list:
    '''
    # emg envelope, trial averaged
    emg_path = (file_name + '_epoched_envelope.parquet')
    pdf_path = folder_path / "figures" / ('epoched_emg_envelope.pdf')
    group_features = ['eid', 'pw']
    relplot_kwargs = dict(
        estimator='mean', errorbar='se', hue='amp', palette='crest')
    emg_dict[file_name] = pd.read_parquet(folder_path / emg_path)
    '''
    # emg, trial averaged
    emg_path = (file_name + '_epoched_emg.parquet')
    pdf_path = folder_path / "figures" / ('epoched_emg.pdf')
    group_features = ['eid', 'pw']
    relplot_kwargs = dict(
        estimator='mean', errorbar='se', hue='amp', palette='crest')
    emg_dict[file_name] = pd.read_parquet(folder_path / emg_path)

emg_df = pd.concat(emg_dict, names=['block'])
del emg_dict

plot_df = emg_df.stack().reset_index().rename(columns={0: 'value'})
plot_df.loc[:, 't_msec'] = plot_df['t'] * 1e3
t_min, t_max = plot_df['t_msec'].min(), plot_df['t_msec'].max()
plot_df['freq'] = plot_df['freq'].astype(int)

if not os.path.exists(folder_path / "figures"):
    os.makedirs(folder_path / "figures")

with PdfPages(pdf_path) as pdf:
    for name, group in plot_df.groupby(group_features):
        these_params = {key: value for key, value in zip(group_features, name)}
        g = sns.relplot(
            data=group,
            row='freq', col='channel',
            x='t_msec', y='value',
            kind='line', height=4, aspect=1,
            facet_kws=dict(sharey=True, margin_titles=True),
            **relplot_kwargs
            )
        for ax in g.axes.flatten():
            ax.set_xlim(-50, 350)
            # ax.set_ylim(-.1, 1.5)
        g.set_titles(col_template="{col_name}", row_template="{row_name} Hz")
        g.set_xlabels('Time (msec.)')
        g.set_ylabels('EMG (mV)')
        g._legend.set_title('Stim. amplitude (uA)')
        g.figure.suptitle(' '.join([f"{key}: {value}" for key, value in these_params.items()]))
        g.figure.align_labels()
        pdf.savefig()
        plt.close()
