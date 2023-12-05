import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler

clinc_sample_rate = 36931.8
'''filterOpts = {
    'high': {
        'Wn': 1000.,
        'N': 2,
        'btype': 'high',
        'ftype': 'butter'
    },
}'''

clinc_sample_interval = pd.Timedelta(27077, unit='ns').to_timedelta64()
clinc_sample_interval_sec = float(clinc_sample_interval) * 1e-9
clinc_sample_rate = (clinc_sample_interval_sec) ** -1

folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/20231109-Phoenix")
# file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555"]
file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555"]

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name_list = [
    'MB_1700670158_174163', 'MB_1700671071_947699', 'MB_1700671568_714180',
    'MB_1700672329_741498', 'MB_1700672668_26337', 'MB_1700673350_780580'
    ]

file_name_list = ['MB_1700672668_26337', 'MB_1700673350_780580']
for file_name in file_name_list:
    print(f"rereferencing {file_name}")
    clinc_df = pd.read_parquet(folder_path / (file_name + '_clinc.parquet'))
    with open(folder_path / 'reref_lookup.json', 'r') as f:
        reref_lookup = json.load(f)[file_name]
    reref_df = pd.DataFrame(0, index=clinc_df.index, columns=[key for key in reref_lookup.keys()])
    for key, value in reref_lookup.items():
        reref_df[key] = clinc_df[key] - clinc_df[value]
    reref_df.to_parquet(folder_path / (file_name + '_clinc_reref.parquet'))
    print('\tDone')


