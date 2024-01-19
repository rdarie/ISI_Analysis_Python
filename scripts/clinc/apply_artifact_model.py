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

run_on_reref = False
per_pulse = False

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
if per_pulse:
    pp_suffix = '_per_pulse'
else:
    pp_suffix = ''

fit_lims = [1e-3, 9e-3]
amp_lims = [1e-3, 2e-3]
offset_lims = [8e-3, 9e-3]
plotting = False

init_ecap_params = Parameters()
init_ecap_params.add('amp1', value=1, vary=True)
init_ecap_params.add(
    'wid1', value=(fit_lims[1] - fit_lims[0]) / 4,
    min=(fit_lims[1] - fit_lims[0]) / 20,
    max=(fit_lims[1] - fit_lims[0]),
    vary=True)
init_ecap_params.add('amp2', value=1, vary=True)
init_ecap_params.add(
    'wid2', value=(fit_lims[1] - fit_lims[0]) / 4,
    min=(fit_lims[1] - fit_lims[0]) / 20,
    max=(fit_lims[1] - fit_lims[0]),
    vary=True)
init_ecap_params.add(
    't_offset1', value=(fit_lims[1] + fit_lims[0]) / 4, min=fit_lims[0], max=fit_lims[1], vary=True)
init_ecap_params.add(
    't_offset2', value=(fit_lims[1] - fit_lims[0]) / 2, min=(fit_lims[1] - fit_lims[0]) / 20,
    max=(fit_lims[1] - fit_lims[0]), vary=True)

init_exp_params = Parameters()
init_exp_params.add('amp', value=1, vary=True)
init_exp_params.add('tau', value=1e-2, vary=True)
init_exp_params.add('offset', value=0, vary=True)

lfp_dict = {}
for file_name in file_name_list:
    lfp_path = (file_name + f'_epoched{file_suffix}_lfp{pp_suffix}.parquet')
    lfp_dict[file_name] = pd.read_parquet(folder_path / lfp_path)

lfp_df = pd.concat(lfp_dict, names=['block'])
lfp_df.sort_index(level='amp', ascending=False, inplace=True)
del lfp_dict

group_features = [nm for nm in lfp_df.index.names if nm != 't']

out_ecaps = {}
out_exps = {}
out_predictions = {}
out_params = {}
for idx, (name, group) in enumerate(tqdm(lfp_df.groupby(group_features))):
    t = group.index.get_level_values('t').to_numpy()
    fit_mask = (t >= fit_lims[0]) & (t <= fit_lims[1])
    amp_mask = (t >= amp_lims[0]) & (t <= amp_lims[1])
    offset_mask = (t >= offset_lims[0]) & (t <= offset_lims[1])
    #
    this_ecap = pd.DataFrame(0, index=t[fit_mask], columns=group.columns)
    this_exp = pd.DataFrame(0, index=t[fit_mask], columns=group.columns)
    this_prediction = pd.DataFrame(0, index=t[fit_mask], columns=group.columns)
    these_params_list = []
    for cn in group.columns:
        artifact_model = ArtifactModel(
            ecap_fun=biphasic_ecap, init_ecap_params=init_ecap_params,
            exp_fun=offset_exponential, init_exp_params=init_exp_params,
            num_macro_iterations=10, verbose=0, fit_opts=dict(method='leastsq')
        )
        y = group.loc[fit_mask, cn].to_numpy()
        #
        artifact_model.exp_params['amp'].set(value=np.abs(group.loc[offset_mask, cn].mean() - group.loc[amp_mask, cn].mean()))
        artifact_model.exp_params['offset'].set(value=group.loc[offset_mask, cn].mean())
        #
        artifact_model.ecap_params['amp1'].set(value=group.loc[fit_mask, cn].max() - group.loc[fit_mask, cn].min())
        artifact_model.ecap_params['amp2'].set(value=group.loc[fit_mask, cn].max() - group.loc[fit_mask, cn].min())
        artifact_model.fit(t[fit_mask], y)
        ecap_fit = artifact_model.ecap_model.eval(t=t[fit_mask], params=artifact_model.ecap_params)
        this_ecap[cn] = ecap_fit
        exp_fit = artifact_model.exp_model.eval(t=t[fit_mask], params=artifact_model.exp_params)
        this_exp[cn] = exp_fit
        this_prediction[cn] = ecap_fit + exp_fit
        these_params_list.append(
            pd.concat(
                {
                    'ecap': pd.DataFrame(artifact_model.ecap_params.valuesdict(), index=[cn]),
                    'exp': pd.DataFrame(artifact_model.exp_params.valuesdict(), index=[cn]),
                },
                axis='columns'
                )
            )
    out_ecaps[name] = this_ecap
    out_exps[name] = this_exp
    out_predictions[name] = this_prediction
    out_params[name] = pd.concat(these_params_list).T
    out_params[name].columns.name = 'channel'
    out_params[name].index.names = ['component', 'parameter']
    if plotting:
        fig, ax = plt.subplots(2, 3)
        flatax = ax.flatten()
        for c_idx, cn in enumerate(group.columns):
            flatax[c_idx].plot(t[fit_mask], group.loc[fit_mask, cn], label='data')
            flatax[c_idx].plot(t[fit_mask], this_exp[cn] + this_ecap[cn], label='fit')
            flatax[c_idx].plot(t[fit_mask], this_ecap[cn], label='ecap')
            flatax[c_idx].plot(t[fit_mask], this_exp[cn], label='exp')
            flatax[c_idx].legend()
            flatax[c_idx].set_title(cn)
        plt.show()

ecaps = pd.concat(out_ecaps, axis='index', names=group_features + ['t'])
ecaps.to_parquet(folder_path / f'artifact_model_ecaps{file_suffix}{pp_suffix}.parquet', engine='fastparquet')
exps = pd.concat(out_exps, axis='index', names=group_features + ['t'])
exps.to_parquet(folder_path / f'artifact_model_exps{file_suffix}{pp_suffix}.parquet', engine='fastparquet')
predictions = pd.concat(out_predictions, axis='index', names=group_features + ['t'])
predictions.to_parquet(folder_path / f'artifact_model_predictions{file_suffix}{pp_suffix}.parquet', engine='fastparquet')
parameters = pd.concat(out_params, axis='index', names=group_features + ['component', 'parameter'])
parameters.to_parquet(folder_path / f'artifact_model_parameters{file_suffix}{pp_suffix}.parquet', engine='fastparquet')

plot_df = predictions.stack().reset_index().rename(columns={0: 'value'})
plot_df.loc[:, 't_msec'] = plot_df['t'] * 1e3
t_min, t_max = plot_df['t_msec'].min(), plot_df['t_msec'].max()

if not os.path.exists(folder_path / "figures"):
    os.makedirs(folder_path / "figures")

pdf_path = folder_path / "figures" / f"artifact_model_predictions{file_suffix}{pp_suffix}.pdf"
relplot_kwargs = dict(
    estimator='mean', errorbar='se', hue='amp', palette='crest')
plot_features = 'eid'
with PdfPages(pdf_path) as pdf:
    for name, group in plot_df.groupby(plot_features):
        data_group = (
            lfp_df.xs(name, level=plot_features)
            .stack().reset_index().rename(columns={0: 'value'}))
        data_group['t_msec'] = data_group['t'] * 1e3
        t = data_group.index.get_level_values('t').to_numpy()
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
