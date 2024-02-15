from pathlib import Path
import pandas as pd
import json, os
import glob

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311071300-Phoenix")
with open(folder_path / 'analysis_metadata/general_metadata.json', 'r') as f:
    general_metadata = json.load(f)
    # file_name_list = general_metadata['file_name_list']
    dsi_block_list = general_metadata['dsi_block_list']

routing_config_info = pd.read_json(folder_path / 'analysis_metadata/routing_config_info.json')
routing_config_info['config_start_time'] = routing_config_info['config_start_time'].apply(lambda x: pd.Timestamp(x, tz='GMT'))
routing_config_info['config_end_time'] = routing_config_info['config_end_time'].apply(lambda x: pd.Timestamp(x, tz='GMT'))

# routing_config_info.sort_values('config_start_time')['child_file_name'].to_list()

all_files_ordered = {}
for parent_file_name, group in routing_config_info.groupby('parent_file_name'):
    file_timestamp_parts = parent_file_name.split('_')
    file_start_time = pd.Timestamp(float(file_timestamp_parts[1]), unit='s', tz='EST')
    file_end_time = group['config_end_time'].iloc[-1].tz_convert('EST')
    for child_file_name in group['child_file_name']:
        all_files_ordered[child_file_name] = [file_start_time, file_end_time]

for file_name in glob.glob(f"{folder_path}/*_f.bin"):
    file_timestamp_parts = file_name.split('_')
    file_start_time = pd.Timestamp(float(file_timestamp_parts[1]), unit='s', tz='EST')
    print(f"{Path(file_name).stem}: starts {file_start_time.strftime('%H:%M:%S')}")
'''
for file_name in routing_config_info['parent_file_name'].unique():
    file_timestamp_parts = file_name.split('_')
    file_start_time = pd.Timestamp(float(file_timestamp_parts[1]), unit='s', tz='EST')
    print(f"{file_name}: starts {file_start_time}")
    all_files_ordered[file_name] = [file_start_time, file_start_time]

for file_name in file_name_list:
    file_timestamp_parts = file_name.split('_')
    file_start_time = pd.Timestamp(float(file_timestamp_parts[1]), unit='s', tz='EST')

    file_contents = pd.read_parquet(folder_path / (file_name + '_clinc_trigs.parquet'))
    file_stop_time = file_start_time + file_contents.index[-1]
    print(f"{file_name}: starts {file_start_time}, ends {file_stop_time.round(freq='L')}")
'''
'''
file_name_list = routing_config_info['child_file_name'].to_list()
for file_name in routing_config_info.sort_values('config_start_time')['child_file_name']:
    file_contents = pd.read_parquet(folder_path / (file_name + '_clinc_trigs.parquet'))
    print(f"{file_name}: starts {file_contents.index[0].strftime('%H:%M:%S')}, ends {file_contents.index[-1].strftime('%H:%M:%S')}")
    all_files_ordered[file_name] = [file_contents.index[0].round(freq='s'), file_contents.index[-1].round(freq='s')]
'''
print('\n')
for dsi_block_name in dsi_block_list:
    file_contents = pd.read_parquet(folder_path / (dsi_block_name + '_dsi_trigs.parquet'))
    print(f"{dsi_block_name}: starts {file_contents.index[0].round(freq='s')}, ends {file_contents.index[-1].round(freq='s')}")
    all_files_ordered[dsi_block_name] = [file_contents.index[0].round(freq='s'), file_contents.index[-1].round(freq='s')]

all_files_ordered_df = pd.DataFrame(all_files_ordered, index=['start', 'stop']).T.sort_values('start')
all_files_ordered_df.loc[:, 'interval'] = all_files_ordered_df.apply(
    lambda x: x['start'].strftime('%H:%M:%S') + ' - ' + x['stop'].strftime('%H:%M:%S'),
    axis='columns')
print(all_files_ordered_df)
