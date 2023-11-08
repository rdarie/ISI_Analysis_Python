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

def apply_rereferencing(
        folder_name, list_of_blocks=[4]):

    # filterOpts = {
    #     'low': {
    #         'Wn': 250.,
    #         'N': 8,
    #         'btype': 'low',
    #         'ftype': 'butter'
    #     },
    # }

    verbose = 0
    data_path = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/3_Preprocessed_Data/{folder_name}")
    parquet_folder = data_path / "parquets"

    ref_chans = ['ch 9 (caudal)', 'ch 17 (caudal)', 'ch 9 (rostral)', 'ch 17 (rostral)']
    implant_map = mapToDF("/users/rdarie/isi_analysis/ISI_Analysis_Python/ripple_map_files/boston_sci.map")
    implant_map.loc[implant_map['elecName'] == 'rostral', 'ycoords'] = implant_map.loc[implant_map['elecName'] == 'rostral', 'ycoords'] + 50000
    implant_map.loc[:, 'lfp_label'] = implant_map.apply(
        lambda row: f"ch {int(row['elecID'])} ({row['elecName']})", axis='columns')

    '''
    plt.scatter(implant_map['xcoords'], implant_map['ycoords'])
    ax = plt.gca()
    for row_idx, row in implant_map.iterrows():
        ax.text(row['xcoords'], row['ycoords'], row['lfp_label'])
    plt.show()
    '''

    '''for row_idx, row in implant_map.iterrows():
        if row_idx == 44:
            distance_to_here = np.sqrt((implant_map['xcoords'] - row['xcoords']) ** 2 + (implant_map['ycoords'] - row['ycoords']) ** 2)
            print(np.unique(distance_to_here))'''

    neighbor_lookup = {}

    for name, group in implant_map.groupby(['elecName', 'xcoords']):
        elecName, xcoords = name
        if elecName == 'rostral':
            pitch_x = 4400
        elif elecName == 'caudal':
            pitch_x = 6600
        for row_idx, row in group.iterrows():
            vertical_distance = (group['ycoords'] - row['ycoords']).abs()
            neighbor_mask = (vertical_distance == pitch_x) & (~ group['lfp_label'].isin(ref_chans))
            if neighbor_mask.sum() == 2:
                neighbor_lookup[row_idx] = group.index[neighbor_mask]
                these_neighbors = [
                    f"ch {int(implant_map.loc[n_idx, 'elecID'])} ({implant_map.loc[n_idx, 'elecName']})"
                    for n_idx in neighbor_lookup[row_idx]]
                print(f"{row['lfp_label']}: {these_neighbors}")

    '''
    distance_mat = pd.DataFrame(0, index=implant_map['lfp_label'], columns=implant_map['lfp_label'])

    for row_idx, row in implant_map.iterrows():
        if row['elecName'] == 'rostral':
            max_distance = 4400
        elif row['elecName'] == 'caudal':
            max_distance = 6600
        distance_to_here = np.sqrt((implant_map['xcoords'] - row['xcoords']) ** 2 + (implant_map['ycoords'] - row['ycoords']) ** 2)
        valid_neighbor_mask = (distance_to_here > 0) & (distance_to_here <= max_distance) & (~ implant_map['lfp_label'].isin(ref_chans))
        neighbor_lookup[row_idx] = implant_map.index[valid_neighbor_mask]
        these_neighbors = [
            f"ch {int(implant_map.loc[n_idx, 'elecID'])} ({implant_map.loc[n_idx, 'elecName']})"
            for n_idx in neighbor_lookup[row_idx]]
        print(f"{row['lfp_label']}: {these_neighbors}")
        distance_mat.loc[row['lfp_label'], :] = distance_to_here.to_numpy()

    np.unique(distance_mat.stack())
    sns.heatmap(distance_mat)
    '''

    '''
    plt.scatter(implant_map.loc[implant_map['elecName'] == 'caudal', 'xcoords'], implant_map.loc[implant_map['elecName'] == 'caudal', 'ycoords'], marker='o')
    plt.scatter(implant_map.loc[implant_map['elecName'] == 'rostral', 'xcoords'], implant_map.loc[implant_map['elecName'] == 'rostral', 'ycoords'], marker='*')
    '''
    lfp_sample_rate = 15000
    # filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), lfp_sample_rate)

    for idx_into_list, block_idx in enumerate(list_of_blocks):
        nf7_parquet_path = parquet_folder / f"Block{block_idx:0>4d}_nf7_df.parquet"
        lfp_df = pd.read_parquet(nf7_parquet_path)
        lfp_df.loc[:, ref_chans] = 0
        reref_df = pd.DataFrame(np.nan, index=lfp_df.index, columns=lfp_df.columns)

        '''
        corr_mat = pd.DataFrame(
            np.corrcoef(lfp_df.iloc[:int(1e6), :].T),
            index=lfp_df.columns, columns=lfp_df.columns)
        sns.heatmap(corr_mat)
        plt.scatter(distance_mat.stack(), corr_mat.fillna(0).stack())
        '''

        '''
        example_index = 12  # caudal 12
        row = implant_map.loc[example_index, :]
        col_name = f"ch {int(row['elecID'])} ({row['elecName']})"
        these_neighbors = [
            f"ch {int(implant_map.loc[n_idx, 'elecID'])} ({implant_map.loc[n_idx, 'elecName']})"
            for n_idx in neighbor_lookup[example_index]]
        this_surround = lfp_df.loc[:, these_neighbors].mean(axis='columns')
        plot_mask = (lfp_df.index > 5.2e8) & (lfp_df.index < 6.2e8)
        
        plt.plot(lfp_df.loc[plot_mask, these_neighbors], label='surround components')

        plt.plot(lfp_df.loc[plot_mask, col_name], label='original')
        plt.plot(this_surround.loc[plot_mask], label='surround')
        plt.plot(lfp_df.loc[plot_mask, col_name] - this_surround.loc[plot_mask], label='difference')
        plt.legend()
        plt.show()
        '''

        for row_idx, neighbors in tqdm(neighbor_lookup.items()):
            row = implant_map.loc[row_idx, :]
            col_name = f"ch {int(row['elecID'])} ({row['elecName']})"
            these_neighbors = [
                f"ch {int(implant_map.loc[n_idx, 'elecID'])} ({implant_map.loc[n_idx, 'elecName']})"
                for n_idx in neighbors]
            print(f"{col_name}: {these_neighbors}")
            this_surround = lfp_df.loc[:, these_neighbors].mean(axis='columns')
            reref_df.loc[:, col_name] = lfp_df.loc[:, col_name] - this_surround
        reref_df = reref_df.loc[:, ~reref_df.isna().all()]

        # reref_df = pd.DataFrame(
        #     signal.sosfiltfilt(filterCoeffs, reref_df - reref_df.mean(), axis=0),
        #     index=reref_df.index, columns=reref_df.columns)

        reref_df.to_parquet(parquet_folder / f"Block{block_idx:0>4d}_reref_lfp_df.parquet")

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
    this_emg_montage = emg_montages['lower_v2']
    list_of_blocks = [5]
    # folder_name = "Day11_PM"
    # list_of_blocks = [2]
    # folder_name = "Day12_AM"
    # list_of_blocks = [3]
    # folder_name = "Day12_PM"
    # list_of_blocks = [4]
    apply_rereferencing(
        folder_name, list_of_blocks=list_of_blocks)
