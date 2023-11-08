
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
import cloudpickle as pickle
import pdb
import os
import gc
import vg
import av
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
from scipy import signal
from scipy.interpolate import UnivariateSpline
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.use('tkagg')   # generate interactive output
useDPI = 200
dpiFactor = 72 / useDPI

from numpy.polynomial import Polynomial
import seaborn as sns
from matplotlib import pyplot as plt

snsRCParams = {
    'figure.dpi': useDPI, 'savefig.dpi': useDPI,
    'lines.linewidth': .5,
    'lines.markersize': 2.5,
    'patch.linewidth': .5,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.linewidth": .125,
    "grid.linewidth": .2,
    "font.size": 4,
    "axes.labelsize": 7,
    "axes.titlesize": 9,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 7,
    "legend.title_fontsize": 9,
    "xtick.bottom": True,
    "xtick.top": False,
    "ytick.left": True,
    "ytick.right": False,
    "xtick.major.width": .125,
    "ytick.major.width": .125,
    "xtick.minor.width": .125,
    "ytick.minor.width": .125,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.minor.size": 1,
    "ytick.minor.size": 1,
    "xtick.direction": 'in',
    "ytick.direction": 'in',
}
mplRCParams = {
    'figure.titlesize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}
sns.set(
    context='talk', style='white',
    palette='dark', font='sans-serif',
    font_scale=2, color_codes=True, rc=snsRCParams
    )
for rcK, rcV in mplRCParams.items():
    mpl.rcParams[rcK] = rcV

from isicpy.utils import mapToDF, makeFilterCoeffsSOS
from isicpy.lookup_tables import emg_montages, kinematics_offsets, video_info

def blank_stim(
        folder_name, list_of_blocks=[4]):

    verbose = 0
    data_path = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/3_Preprocessed_Data/{folder_name}")
    parquet_folder = data_path / "parquets"

    ref_chans = ['ch 9 (caudal)', 'ch 17 (caudal)', 'ch 9 (rostral)', 'ch 17 (rostral)']

    for idx_into_list, block_idx in enumerate(list_of_blocks):
        nf7_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_nf7_df.parquet"
        print("#" * 20 + f"\nLoading {nf7_parquet_path}\n" + "#" * 20)
        lfp_df = pd.read_parquet(nf7_parquet_path)
        lfp_df.drop(columns=ref_chans, inplace=True)

        nev_spikes_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_nev_spikes_df.parquet"
        nev_df = pd.read_parquet(nev_spikes_parquet_path)

        stim_times = nev_df['time_usec'].drop_duplicates()
        next_times = pd.Series(stim_times.shift(-1).fillna(-1).astype(int).to_numpy(), index=stim_times)
        next_times.iloc[-1] = stim_times.iloc[-1] + int(100e-3)

        # crop for debugging
        '''
        db_left, db_right = stim_times[0] - 1000, stim_times[0] + int(20 * 1e6)
        db_mask = (lfp_df.index > db_left) & (lfp_df.index < db_right)
        lfp_df = lfp_df.loc[db_mask, :]
        nev_df = nev_df.loc[nev_df['time_usec'] < db_right, :]
        next_times = next_times.loc[next_times.index < db_right]
        '''

        blank_width = 1250  # usec

        for st, nev_group in tqdm(nev_df.groupby('time_usec')):
            spike_mask = (lfp_df.index > st) & (lfp_df.index <= next_times[st])
            for col_name in lfp_df.columns:
                # fig, ax = plt.subplots()
                # col_name = "ch 18 (caudal)"
                this_signal = lfp_df.loc[spike_mask, col_name].copy()
                # ax.plot(this_signal, 'b-', lw=1)
                spline_mask = this_signal.index > st + 2 * blank_width
                spl = UnivariateSpline(this_signal.index[spline_mask], this_signal.loc[spline_mask], k=1)
                fill_mask = this_signal.index > st + blank_width
                this_signal.loc[fill_mask] = this_signal.loc[fill_mask] - spl(this_signal.index[fill_mask])
                #
                blank_mask = (this_signal.index > st) & (this_signal.index <= st + blank_width)
                this_signal.loc[blank_mask] = np.nan
                #
                # ax.plot(this_signal.index, spl(this_signal.index))
                # ax.plot(this_signal)
                lfp_df.loc[spike_mask, col_name] = this_signal.to_numpy()
            # plot_mask = (lfp_df.index > st - 1000) & (lfp_df.index <= next_times[st])
            # plt.plot(lfp_df.loc[plot_mask, 'ch 18 (caudal)'])
        lfp_df.interpolate(inplace=True, method='linear')
        print(f"lfp_df.isna().any(): {lfp_df.isna().any()}")
        lfp_df.to_parquet(parquet_folder / f"Block{block_idx:0>4d}_stim_blanked_lfp_df.parquet")
    return


if __name__ == '__main__':
    # folder_name = "Day7_AM"
    # list_of_blocks = [4]
    # folder_name = "Day8_AM"
    # list_of_blocks = [3]
    # folder_name = "Day8_PM"
    # list_of_blocks = [2]
    # folder_name = "Day11_AM"
    # list_of_blocks = [2]
    folder_name = "Day11_PM"
    list_of_blocks = [2, 3]
    # this_emg_montage = emg_montages['lower_v2']
    # block_idx_list = [2, 3]
    # folder_name = "Day11_PM"
    # list_of_blocks = [2]
    # folder_name = "Day12_AM"
    # list_of_blocks = [3]
    # folder_name = "Day12_PM"
    # list_of_blocks = [4]
    blank_stim(
        folder_name, list_of_blocks=list_of_blocks)
