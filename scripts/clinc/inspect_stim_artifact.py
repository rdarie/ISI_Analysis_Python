import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output

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
    context='notebook', style='darkgrid',
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

def plot_single_event(data, channel=None):
    plot_data = data.query("channel == 'S1_S3'")
    plt.plot(plot_data['t_msec'], plot_data['value'])
    plt.show()
    return

# folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/20231109-Phoenix")
# file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555"]

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name_list = [
    'MB_1700670158_174163', 'MB_1700671071_947699', 'MB_1700671568_714180',
    'MB_1700672329_741498', 'MB_1700672668_26337', 'MB_1700673350_780580'
    ]

file_name_list = ['MB_1700672668_26337']

for file_name in file_name_list:
    lfp_df = pd.read_parquet(folder_path / (file_name + '_epoched_reref_lfp.parquet'))
    # plot_df = lfp_df.stack().reset_index().rename(columns={0: 'value'})
    plot_df = lfp_df.reset_index()
    plot_df.loc[:, 't_msec'] = plot_df['t'] * 1e3
    
    for name, group in plot_df.groupby(['timestamp']):
        chans = lfp_df.columns
        # chans = ['E0', 'E47']
        fig, ax = plt.subplots()
        for cn in chans:
            ax.plot(group['t_msec'], group[cn], label=cn)
        ax.set_xlim([2, 9])
        ax.legend()
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


