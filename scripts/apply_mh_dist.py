
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.use('QT5Agg')   # generate interactive output

import traceback
from isicpy.utils import mapToDF, makeFilterCoeffsSOS
from isicpy.lookup_tables import emg_montages, kinematics_offsets, video_info
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
import ephyviewer
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
from sklearn.covariance import MinCovDet

def apply_mh_dist(
        folder_name, list_of_blocks=[4]):

    verbose = 0
    data_path = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/3_Preprocessed_Data/{folder_name}")
    parquet_folder = data_path / "parquets"

    ref_chans = ['ch 9 (caudal)', 'ch 17 (caudal)', 'ch 9 (rostral)', 'ch 17 (rostral)']

    for idx_into_list, block_idx in enumerate(list_of_blocks):
        nf7_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_nf7_df.parquet"
        lfp_df = pd.read_parquet(nf7_parquet_path)
        lfp_df.drop(columns=ref_chans, inplace=True)

        # fit a MCD robust estimator to data
        robust_cov = MinCovDet().fit(lfp_df.iloc[:int(1e5), :])

        output_df = pd.DataFrame(robust_cov.mahalanobis(lfp_df), index=lfp_df.index, columns=['mahalanobis_distance'])
        output_df.to_parquet(parquet_folder / f"Block{block_idx:0>4d}_mhd_df.parquet")

    return


if __name__ == '__main__':
    # folder_name = "Day7_AM"
    # list_of_blocks = [4]
    # folder_name = "Day8_AM"
    # list_of_blocks = [3]
    folder_name = "Day8_PM"
    list_of_blocks = [2]
    # folder_name = "Day11_AM"
    # list_of_blocks = [2]
    folder_name = "Day11_PM"
    this_emg_montage = emg_montages['lower_v2']
    list_of_blocks = [5]
    # folder_name = "Day11_PM"
    # list_of_blocks = [2]
    # folder_name = "Day12_AM"
    # list_of_blocks = [3]
    # folder_name = "Day12_PM"
    # list_of_blocks = [4]
    apply_mh_dist(
        folder_name, list_of_blocks=list_of_blocks)
