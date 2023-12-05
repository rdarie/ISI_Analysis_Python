import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
import os
import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set(
    context='notebook', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True,
    )

from matplotlib import pyplot as plt
clinc_sample_rate = 36931.8
'''filterOpts = {
    'high': {
        'Wn': 1000.,
        'N': 2,
        'btype': 'high',
        'ftype': 'butter'
    },
}'''

clinc_sample_interval = pd.Timedelta(27077, unit='ns').to_timedelta64()
clinc_sample_interval_sec = float(clinc_sample_interval) * 1e-9
clinc_sample_rate = clinc_sample_interval_sec ** -1

folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/20231109-Phoenix")
# file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555"]
file_name_list = ["MB_1699560317_650555"]

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name_list = [
    'MB_1700670158_174163', 'MB_1700671071_947699', 'MB_1700671568_714180',
    'MB_1700672329_741498', 'MB_1700672668_26337', 'MB_1700673350_780580'
    ]

file_name_list = ['MB_1700671568_714180']

for file_name in file_name_list:

    lfp_path = (file_name + '_epoched_lfp.parquet')
    pdf_path = folder_path / "figures" / (file_name + '_epoched_lfp.pdf')
    group_features = ['eid', 'pw']
    relplot_kwargs = dict(estimator='mean', errorbar='se', hue='amp')

    '''
    lfp_path = (file_name + '_epoched_lfp.parquet')
    pdf_path = folder_path / "figures" / (file_name + '_epoched_lfp_single.pdf')
    group_features = ['eid', 'pw', 'amp']
    relplot_kwargs = dict(estimator=None, units='timestamp')
    '''

    '''
    lfp_path = (file_name + '_epoched_reref_lfp.parquet')
    pdf_path = folder_path / "figures" / (file_name + '_epoched_reref_lfp.pdf')
    group_features = ['eid', 'pw']
    relplot_kwargs = dict(estimator='mean', errorbar='se', hue='amp')
    '''

    '''
    lfp_path = (file_name + '_epoched_reref_lfp.parquet')
    pdf_path = folder_path / "figures" / (file_name + '_epoched_reref_lfp_single.pdf')
    group_features = ['eid', 'pw', 'amp']
    relplot_kwargs = dict(estimator=None, units='timestamp')
    '''

    lfp_df = pd.read_parquet(folder_path / lfp_path)
    plot_df = lfp_df.stack().reset_index().rename(columns={0: 'value'})
    plot_df.loc[:, 't_msec'] = plot_df['t'] * 1e3
    t_min, t_max = plot_df['t_msec'].min(), plot_df['t_msec'].max()
    dz0 = 1e-2  # 10 us min
    if not os.path.exists(folder_path / "figures"):
        os.makedirs(folder_path / "figures")
    with PdfPages(pdf_path) as pdf:
        for name, group in plot_df.groupby(group_features):
            if 'amp' in group_features:
                eid, pw, amp = name
            else:
                eid, pw = name
            g = sns.relplot(
                data=group,
                col='channel', col_wrap=6,
                x='t_msec', y='value',
                kind='line',
                facet_kws=dict(sharey=False),
                **relplot_kwargs
                )
            for ax in g.axes.flatten():
                ax.set_xlim(t_min, 2)
                for line_t in [pw * 1e-3 + dz0, pw * 5e-3 + dz0]:
                    ax.axvline(line_t, color='r')
            if 'amp' in group_features:
                g.figure.suptitle(f'amp: {amp} uA')
            pdf.savefig()
            plt.close()
