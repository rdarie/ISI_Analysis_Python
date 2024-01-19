import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from isicpy.clinc_lookup_tables import clinc_sample_rate, emg_sample_rate, dsi_trig_sample_rate
from isicpy.artifact_model import *
from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from lmfit.models import GaussianModel, ExponentialModel
from matplotlib.backends.backend_pdf import PdfPages
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set(
    context='notebook', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True,
    )

from matplotlib import pyplot as plt

clinc_sample_rate = 36931.8
run_on_reref = True
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

if run_on_reref:
    file_suffix = '_reref'
else:
    file_suffix = ''
fit_lims = [1e-3, 9e-3]
amp_lims = [1e-3, 2e-3]
offset_lims = [8e-3, 9e-3]
plotting = False

lfp_dict = {}
for file_name in file_name_list:
    lfp_path = (file_name + f'_epoched{file_suffix}_lfp.parquet')
    lfp_dict[file_name] = pd.read_parquet(folder_path / lfp_path)

lfp_df = pd.concat(lfp_dict, names=['block'])
lfp_df.sort_index(level='amp', ascending=False, inplace=True)
del lfp_dict

group_features = [nm for nm in lfp_df.index.names if nm != 't']

ecaps = pd.read_parquet(folder_path / f'artifact_model_ecaps{file_suffix}.parquet')
exps = pd.read_parquet(folder_path / f'artifact_model_exps{file_suffix}.parquet')
predictions = pd.read_parquet(folder_path / f'artifact_model_predictions{file_suffix}.parquet')
parameters = pd.read_parquet(folder_path / f'artifact_model_parameters{file_suffix}.parquet')

plot_df = predictions.stack().reset_index().rename(columns={0: 'value'})
plot_df.loc[:, 't_msec'] = plot_df['t'] * 1e3
t_min, t_max = plot_df['t_msec'].min(), plot_df['t_msec'].max()

if not os.path.exists(folder_path / "figures"):
    os.makedirs(folder_path / "figures")

pdf_path = folder_path / "figures" / f"artifact_model_predictions{file_suffix}.pdf"
relplot_kwargs = dict(
    estimator='mean', errorbar='se', hue='amp', palette='crest')
plot_features = 'eid'
with PdfPages(pdf_path) as pdf:
    for name, group in plot_df.groupby(plot_features):
        data_group = (
            lfp_df.xs(name, level=plot_features)
            .stack().reset_index().rename(columns={0: 'value'}))
        data_group['t_msec'] = data_group['t'] * 1e3
        t = data_group['t'].to_numpy()
        fit_mask = (t >= fit_lims[0]) & (t <= fit_lims[1])
        g = sns.relplot(
            data=data_group.loc[fit_mask, :],
            col='channel', col_wrap=6,
            x='t_msec', y='value',
            kind='line',
            facet_kws=dict(sharey=False),
            **relplot_kwargs
            )
        for ax in g.axes.flatten():
            ax.set_xlim(1, t_max)
        pdf.savefig()
        plt.show()
        #
        g = sns.relplot(
            data=group,
            col='channel', col_wrap=6,
            x='t_msec', y='value',
            kind='line',
            facet_kws=dict(sharey=False),
            **relplot_kwargs
            )
        for ax in g.axes.flatten():
            ax.set_xlim(1, t_max)
        pdf.savefig()
        plt.show()
        #
        ecap_group = (
            ecaps.xs(name, level=plot_features)
            .stack().reset_index().rename(columns={0: 'value'}))
        ecap_group['t_msec'] = ecap_group['t'] * 1e3
        g = sns.relplot(
            data=ecap_group,
            col='channel', col_wrap=6,
            x='t_msec', y='value',
            kind='line',
            facet_kws=dict(sharey=False),
            **relplot_kwargs
            )
        for ax in g.axes.flatten():
            ax.set_xlim(1, t_max)
        pdf.savefig()
        plt.show()
