import os
from isicpy.utils import load_synced_mat, closestSeries
from isicpy.lookup_tables import emg_montages
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pdb
import traceback
from tqdm import tqdm
import cloudpickle as pickle
from sklearn.preprocessing import StandardScaler

this_emg_montage = emg_montages['lower']
# data_path = Path("/users/rdarie/scratch/3_Preprocessed_Data/Day12_PM")
# blocks_list = [1, 2, 3, 4]
# data_path = Path("/users/rdarie/scratch/3_Preprocessed_Data/Day11_PM")
# blocks_list = [2, 3]
# data_path = Path("/users/rdarie/scratch/3_Preprocessed_Data/Day8_AM")
# blocks_list = [3, 4]
# data_path = Path("/users/rdarie/scratch/3_Preprocessed_Data/Day12_AM")
# blocks_list = [2, 3]
data_path = Path("/users/rdarie/scratch/3_Preprocessed_Data/Day2_AM")
blocks_list = [3]

all_emg = {}
for block_idx in tqdm(blocks_list):
    file_path = data_path / f"Block{block_idx:0>4d}_Synced_Session_Data.mat"
    data_dict = load_synced_mat(
        file_path, load_vicon=True, vicon_as_df=True)
    if data_dict['vicon'] is not None:
        if 'EMG' in data_dict['vicon']:
            all_emg[block_idx] = data_dict['vicon']['EMG'].copy()
            all_emg[block_idx].rename(columns=this_emg_montage, inplace=True)
            all_emg[block_idx].drop(columns=['NA'], inplace=True)

all_emg_df = pd.concat(all_emg)
scaler = StandardScaler()
scaler.fit(all_emg_df)

if not os.path.exists(data_path / "pickles"):
    os.mkdir(data_path / "pickles")

output_path = data_path / "pickles" / "emg_scaler.p"
with open(output_path, 'wb') as handle:
    pickle.dump(scaler, handle)

