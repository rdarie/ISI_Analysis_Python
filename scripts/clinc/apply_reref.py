import pandas as pd
from pathlib import Path
import json
from isicpy.clinc_lookup_tables import clinc_sample_rate, sid_to_intan, emg_sample_rate, dsi_trig_sample_rate

'''filterOpts = {
    'high': {
        'Wn': 1000.,
        'N': 2,
        'btype': 'high',
        'ftype': 'butter'
    },
}'''
'''
folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311071300-Phoenix")
file_name_list = [
    "MB_1699382682_316178", "MB_1699383052_618936", "MB_1699383757_778055", "MB_1699384177_953948",
    "MB_1699382925_691816", "MB_1699383217_58381", "MB_1699383957_177840"
    ]
'''
'''
folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/202311091300-Phoenix")
file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555"]
'''

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name_list = [
    'MB_1700670158_174163', 'MB_1700671071_947699', 'MB_1700671568_714180',
    'MB_1700672329_741498', 'MB_1700672668_26337', 'MB_1700673350_780580'
    ]

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")
file_name_list = [
    "MB_1702047397_450767", "MB_1702048897_896568", "MB_1702049441_627410",
    "MB_1702049896_129326", "MB_1702050154_688487", "MB_1702051241_224335"
]
file_name_list = [
    "MB_1702050154_688487", "MB_1702051241_224335"
]

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")
file_name_list = ["MB_1702049441_627410", "MB_1702049896_129326"]

for file_name in file_name_list:
    print(f"rereferencing {file_name}")
    clinc_df = pd.read_parquet(folder_path / (file_name + '_clinc.parquet'))
    with open(folder_path / 'reref_lookup.json', 'r') as f:
        reref_lookup = json.load(f)[file_name]
    reref_df = pd.DataFrame(0, index=clinc_df.index, columns=[key for key in reref_lookup.keys()])
    for key, value in reref_lookup.items():
        reref_df[key] = clinc_df[key] - clinc_df[value]
    reref_df.to_parquet(folder_path / (file_name + '_clinc_reref.parquet'), engine='fastparquet')
    print('\tDone')
