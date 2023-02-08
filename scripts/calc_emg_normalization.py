from isicpy.utils import load_synced_mat, closestSeries
from isicpy.lookup_tables import emg_montages
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pdb
import traceback
import cloudpickle as pickle
from sklearn.preprocessing import StandardScaler

left_sweep = -0.1
right_sweep = 0.3
this_emg_montage = emg_montages['lower']
all_emg = {}

for block_idx in [2, 3]:
    data_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/3_Preprocessed_Data/Day11_PM")
    file_path = data_path / f"Block{block_idx:0>4d}_Synced_Session_Data.mat"
    data_dict = load_synced_mat(
        file_path,
        load_vicon=True, vicon_as_df=True,
        )

    if data_dict['vicon'] is not None:
        all_emg[block_idx] = data_dict['vicon']['EMG'].iloc[:, :12].copy()
        all_emg[block_idx].columns = this_emg_montage
        all_emg[block_idx].columns.name = 'label'

all_emg_df = pd.concat(all_emg)
scaler = StandardScaler()
scaler.fit(all_emg_df)

output_path = data_path / "pickles" / "emg_scaler.p"
with open(output_path, 'wb') as handle:
    pickle.dump(scaler, handle)


