from isicpy.third_party.pymatreader import hdf5todict
from isicpy.utils import makeFilterCoeffsSOS
from isicpy.clinc_lookup_tables import clinc_sample_rate, sid_to_intan, emg_sample_rate, dsi_trig_sample_rate
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import signal
import json, yaml, os

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311071300-Phoenix")
folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/202311091300-Phoenix")
folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")

file_path_list = list(folder_path.glob('*.csv')) + list(folder_path.glob('*.ascii'))
file_path_list = [
    fn for fn in file_path_list
    if '_log' not in fn.name
]

start_times = {}
for dsi_block_path in file_path_list:
    print(dsi_block_path)
    dsi_df = pd.read_csv(
        dsi_block_path, header=12, index_col=0, low_memory=False, nrows=1)
    start_times[dsi_block_path.stem] = dsi_df.index[0]

output_times = pd.Series(start_times)
output_times.to_json(folder_path / 'dsi_start_times.json')
