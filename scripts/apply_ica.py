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
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    mpl.use('Agg')   # generate postscript output
else:
    mpl.use('QT5Agg')   # generate interactive output
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

def apply_ica(
        folder_name, list_of_blocks=[4]):

    verbose = 0
    data_path = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/3_Preprocessed_Data/{folder_name}")
    parquet_folder = data_path / "parquets"

    ref_chans = ['ch 9 (caudal)', 'ch 17 (caudal)', 'ch 9 (rostral)', 'ch 17 (rostral)']

    for idx_into_list, block_idx in enumerate(list_of_blocks):
        nf7_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_nf7_df.parquet"
        lfp_df = pd.read_parquet(nf7_parquet_path)
        lfp_df.drop(columns=ref_chans, inplace=True)

        fit_mask = (lfp_df.index > 50e6) & (lfp_df.index < 100e6)

        from sklearn.decomposition import FastICA

        sources = {}
        ica_dict = {}
        for paddle in ['caudal', 'rostral']:
            paddle_mask = lfp_df.columns.str.contains(paddle)
            this_lfp = lfp_df.loc[:, paddle_mask]
            # Compute ICA
            ica_dict[paddle] = FastICA(n_components=this_lfp.shape[1], whiten="arbitrary-variance")
            ica_dict[paddle].fit(this_lfp.loc[fit_mask, :])
            col_names = [f"ica {idx} ({paddle})" for idx in range(this_lfp.shape[1])]
            sources[paddle] = pd.DataFrame(
                ica_dict[paddle].transform(this_lfp), index=this_lfp.index, columns=col_names)  # Reconstruct signals
            # A_ = ica.mixing_  # Get estimated mixing matrix
        ###
        sources_df = pd.concat([value for key, value in sources.items()], axis='columns')
        sources_df.to_parquet(parquet_folder / f"Block{block_idx:0>4d}_ica_sources_df.parquet")
        ###
        for paddle in ['caudal', 'rostral']:
            fig, ax = plt.subplots(5, 6, sharex=True, sharey=True)
            flat_axes = ax.flatten()
            for idx, this_ax in enumerate(flat_axes):
                flat_axes[idx].plot(sources[paddle].iloc[:100000, idx])
                flat_axes[idx].set_title(f"{idx}")
            fig.suptitle(paddle)
            plt.show()
        ###
        sources['caudal'].iloc[:, [9, 10]] = 0.

        recon_list = []
        for paddle in ['caudal', 'rostral']:
            paddle_mask = lfp_df.columns.str.contains(paddle)
            recon_list.append(pd.DataFrame(
                ica_dict[paddle].inverse_transform(sources[paddle]),
                index=this_lfp.index, columns=lfp_df.columns[paddle_mask]))

        reref_df = pd.concat(recon_list, axis='columns')
        reref_df.to_parquet(parquet_folder / f"Block{block_idx:0>4d}_ica_df.parquet")

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
    # folder_name = "Day11_PM"
    # this_emg_montage = emg_montages['lower_v2']
    # block_idx_list = [2, 3]
    # folder_name = "Day11_PM"
    # list_of_blocks = [2]
    # folder_name = "Day12_AM"
    # list_of_blocks = [3]
    # folder_name = "Day12_PM"
    # list_of_blocks = [4]
    apply_rereferencing(
        folder_name, list_of_blocks=list_of_blocks)
