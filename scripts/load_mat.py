from scipy.io import loadmat
import pandas as pd
from pathlib import Path
from pymatreader import read_mat

data_path = Path("E:\\Neural Recordings\\raw\\ISI-C-003\\3_Preprocessed_Data\\Day11_PM")
file_path = data_path / "Block0001_Synced_Session_Data.mat"
try:
    mat_dict = loadmat(file_path)
except NotImplementedError as e:
    mat_dict = read_mat(file_path)

print(mat_dict['Synced_Session_Data']['Ripple_Data']['NEV']['Data']['Spikes'])