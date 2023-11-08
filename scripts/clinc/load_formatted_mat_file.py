from isicpy.third_party.pymatreader import hdf5todict
from pathlib import Path
import h5py

folder_path = Path(r"F:\Neural Recordings\raw\20231107-Phoenix\formatted")
file_path = folder_path / "MB_1699382925_691816_f.mat"

with h5py.File(file_path, 'r') as hdf5_file:
    data = hdf5todict(hdf5_file, variable_names=['data_this_file'], ignore_fields=None)
