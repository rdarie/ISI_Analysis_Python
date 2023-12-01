import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output

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
clinc_sample_rate = (clinc_sample_interval_sec) ** -1

def plot_single_event(data, channel=None):
    plot_data = data.query("channel == 'S1_S3'")
    plt.plot(plot_data['t_msec'], plot_data['value'])
    plt.show()
    return

folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/20231109-Phoenix")
# file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555"]
file_name_list = ["MB_1699558933_985097"]
for file_name in file_name_list:
    lfp_df = pd.read_parquet(folder_path / (file_name + '_epoched_lfp.parquet'))
    # plot_df = lfp_df.stack().reset_index().rename(columns={0: 'value'})
    plot_df = lfp_df.reset_index()
    plot_df.loc[:, 't_msec'] = plot_df['t'] * 1e3
    
    for name, group in plot_df.groupby(['timestamp']):
        chans = lfp_df.columns
        # chans = ['S1_S3']
        for cn in chans:
            plt.plot(group['t_msec'], group[cn], label=cn)
        plt.legend()
        plt.show()
        break

    artifact_df = pd.read_parquet(folder_path / (file_name + '_epoched_artifact.parquet'))
    plot_df = artifact_df.reset_index()
    plot_df.loc[:, 't_msec'] = plot_df['t'] * 1e3

    for name, group in plot_df.groupby(['timestamp']):
        chans = artifact_df.columns
        for cn in chans:
            plt.plot(group['t_msec'], group[cn], label=cn)
        plt.legend()
        plt.show()
        break


