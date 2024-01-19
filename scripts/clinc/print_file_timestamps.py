from pathlib import Path
import pandas as pd

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name_list = [
    'MB_1700670158_174163', 'MB_1700671071_947699', 'MB_1700671568_714180',
    'MB_1700672329_741498', 'MB_1700672668_26337', 'MB_1700673350_780580'
    ]

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311071300-Phoenix")
file_name_list = [
    "MB_1699382682_316178", "MB_1699383052_618936", "MB_1699383757_778055", "MB_1699384177_953948",
    "MB_1699382925_691816", "MB_1699383217_58381", "MB_1699383957_177840"
]

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")
file_name_list = [
    "MB_1702047397_450767",  "MB_1702048897_896568",  "MB_1702049441_627410",
    "MB_1702049896_129326",  "MB_1702050154_688487",  "MB_1702051241_224335"
]
for file_name in file_name_list:
    file_timestamp_parts = file_name.split('_')
    file_start_time = pd.Timestamp(float(file_timestamp_parts[1]), unit='s', tz='EST')
    print(f"{file_name}: starts {file_start_time}")

for file_name in file_name_list:
    file_timestamp_parts = file_name.split('_')
    file_start_time = pd.Timestamp(float(file_timestamp_parts[1]), unit='s', tz='EST')

    file_contents = pd.read_parquet(folder_path / (file_name + '_clinc_trigs.parquet'))
    file_stop_time = file_start_time + file_contents.index[-1]
    print(f"{file_name}: starts {file_start_time}, ends {file_stop_time.round(freq='L')}")
